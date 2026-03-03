from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.formatter import OutputFormatter
from app.agent.nlu import RuleBasedNLU
from app.agent.planner import Planner
from app.agent.ranker import Ranker
from app.agent.state import StateStore
from app.clients.exceptions import DataSourceError
from app.clients.houses import HousesClient
from app.infra.cache import CacheManager
from app.infra.logging import log_event, preview_text
from app.schemas import (
    HardConstraints,
    HouseLite,
    IntentType,
    InvokeRequest,
    InvokeResponse,
    Listing,
    Platform,
    SearchSnapshot,
    SessionPhase,
    SessionState,
    SoftPreferences,
    StructuredQuery,
    TurnSummary,
)

LOGGER = logging.getLogger(__name__)

# Explicit ordinal only (e.g., 第一套/第2套). Do not match “这一套”.
_RANK_INDEX_PATTERN = re.compile(r"第\s*([一二两三四五六七八九123456789])\s*套")
_CH_NUM_TO_INT = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_REPLACE_SIGNALS = ("换", "改成", "换成", "重新", "重来")
_HOUSE_REF_WORDS = ("这套", "这一套", "这间", "这个房", "它", "上一套", "刚才那套", "最开始", "最初")
_EXPLICIT_HOUSE_ID_PATTERN = re.compile(r"([A-Z]{2,4}_?\d{1,8})", re.IGNORECASE)
_COMPLAINT_SIGNALS = (
    "住的不舒服",
    "不太舒服",
    "不舒服",
    "采光不好",
    "太小",
    "房间小",
    "阴暗",
    "通勤太长",
    "噪音大",
)
_SEARCH_SIGNALS = ("找房", "找个房", "推荐", "筛选", "查询", "搜", "帮我找", "看看房", "有房", "房源")
_TOOL_OPERATION_TO_INTENT = {
    "get_houses_by_platform": IntentType.search,
    "get_houses_by_community": IntentType.search,
    "get_houses_nearby": IntentType.search,
    "get_house_by_id": IntentType.house_detail,
    "get_house_listings": IntentType.listings,
    "rent_house": IntentType.rent,
    "terminate_rental": IntentType.terminate,
    "take_offline": IntentType.offline,
    "get_nearby_landmarks": IntentType.amenities,
}
_TOOL_ARGUMENT_ALIASES = {
    "min_price": "budget_min",
    "max_price": "budget_max",
    "rental_type": "rent_type",
    "min_area": "area_min",
    "available_from_before": "move_in_date",
    "commute_to_xierqi_max": "max_commute_min",
    "landmark_id": "landmark_id",
    "max_distance": "max_distance",
}


class DialogueManager:
    def __init__(
        self,
        *,
        state_store: StateStore,
        nlu: RuleBasedNLU,
        planner: Planner,
        ranker: Ranker,
        formatter: OutputFormatter,
        houses_client: HousesClient,
        cache: CacheManager,
        max_output_candidates: int,
    ) -> None:
        self.state_store = state_store
        self.nlu = nlu
        self.planner = planner
        self.ranker = ranker
        self.formatter = formatter
        self.houses_client = houses_client
        self.cache = cache
        self.max_output_candidates = max_output_candidates

    async def handle_turn(self, request: InvokeRequest, state: SessionState, is_new_session: bool) -> InvokeResponse:
        log_event(
            LOGGER,
            "dialogue.turn.start",
            is_new_session=is_new_session,
            phase=state.phase.value,
            message=preview_text(request.message, limit=200),
        )
        if is_new_session:
            await self._init_session_data(state)

        query = self._build_query(request, state)
        merged = self._merge_query_with_state(query, state, request.message)
        if merged.intent in {IntentType.house_detail, IntentType.listings, IntentType.rent, IntentType.terminate, IntentType.offline}:
            explicit_house_id = self._extract_explicit_house_id(request.message)
            if explicit_house_id:
                merged.hard.house_id = explicit_house_id
            else:
                resolved_house_id = self._resolve_house_id(request.message, merged, state)
                if merged.hard.house_id is None and resolved_house_id:
                    merged.hard.house_id = resolved_house_id
        if self._should_treat_search_as_chat(request.message, merged):
            log_event(
                LOGGER,
                "dialogue.intent.adjusted",
                original_intent=IntentType.search.value,
                adjusted_intent=IntentType.chat.value,
                reason="complaint_without_explicit_search_request",
            )
            merged.intent = IntentType.chat
        log_event(
            LOGGER,
            "dialogue.nlu.done",
            intent=merged.intent.value,
            hard=merged.hard.model_dump(exclude_none=True),
            soft=merged.soft.model_dump(exclude_none=True),
            clarify_count=len(merged.clarify_questions),
        )

        try:
            if merged.intent in {IntentType.rent, IntentType.terminate, IntentType.offline}:
                log_event(LOGGER, "dialogue.action.detected", intent=merged.intent.value)
                resp = await self._handle_action(merged, state)
            elif merged.intent == IntentType.listings:
                resp = await self._handle_listings_query(merged, state)
            elif merged.intent == IntentType.house_detail:
                resp = await self._handle_house_detail(merged, state)
            elif merged.intent == IntentType.search and state.phase in {
                SessionPhase.chatting,
                SessionPhase.slot_filling,
            }:
                # In evaluation mode, avoid clarification templates and execute with current constraints.
                resp = await self._handle_search(merged, request, state)
            elif merged.intent == IntentType.chat:
                log_event(LOGGER, "dialogue.chat")
                self._remember_chat_preferences(state, merged)
                state.phase = SessionPhase.chatting
                assistant_reply = self._extract_llm_assistant_reply(request.meta)
                resp = InvokeResponse(
                    text=assistant_reply or self._fallback_chat_reply(request.message, merged),
                    debug={"response_kind": "chat"},
                )
            else:
                resp = await self._handle_search(merged, request, state)
        except DataSourceError as exc:
            log_event(LOGGER, "dialogue.error", error=str(exc))
            LOGGER.exception("Data source error")
            resp = InvokeResponse(text=f"接口调用失败：{exc}，请稍后重试。", candidates=[], debug={"response_kind": "error"})

        self._remember_turn(state, request.message, resp, merged.intent)
        self.state_store.upsert(state)
        return resp

    def _build_query(self, request: InvokeRequest, state: SessionState) -> StructuredQuery:
        llm_parse = request.meta.get("llm_parse")
        if isinstance(llm_parse, dict) and self._llm_parse_has_signal(llm_parse):
            # Start at zero confidence so low-signal parses can cleanly fall back to rule NLU.
            llm_query = StructuredQuery(confidence=0.0)
            llm_query = self._apply_llm_parse(llm_query, llm_parse)
            return llm_query
        # Fallback only when model parse is missing/invalid.
        return self.nlu.parse(request.message, state, request.case_type)

    @staticmethod
    def _llm_parse_has_signal(llm_parse: dict[str, Any]) -> bool:
        for key in ("intent", "tool_plan", "hard", "soft"):
            if key in llm_parse:
                return True
        return False

    @staticmethod
    def _has_explicit_search_request(text: str) -> bool:
        return any(token in text for token in _SEARCH_SIGNALS)

    @staticmethod
    def _has_complaint_signal(text: str) -> bool:
        return any(token in text for token in _COMPLAINT_SIGNALS)

    def _should_treat_search_as_chat(self, text: str, query: StructuredQuery) -> bool:
        if query.intent != IntentType.search:
            return False
        if self._has_explicit_search_request(text):
            return False
        if not self._has_complaint_signal(text):
            return False
        return self._has_actionable_preferences(query)

    async def _handle_search(self, query: StructuredQuery, request: InvokeRequest, state: SessionState) -> InvokeResponse:
        state.phase = SessionPhase.searching
        plan = await self.planner.build_plan(query)
        log_event(LOGGER, "dialogue.plan.built", plan_type=plan.plan_type, landmark_id=plan.landmark_id)
        candidates = await self.planner.execute_plan(plan, query, request.case_type)
        log_event(LOGGER, "dialogue.plan.executed", candidate_count=len(candidates))
        top = await self.ranker.rank_two_stage(candidates, query, max_output=self.max_output_candidates)
        log_event(LOGGER, "dialogue.rank.done", top_count=len(top))

        state.phase = SessionPhase.presenting
        state.confirmed_constraints = query.hard
        state.soft_preferences = query.soft
        state.last_candidates = self._compress_candidates(candidates)
        state.last_top5 = top
        self._remember_search_snapshot(state, request.message, query)

        return self.formatter.render(
            case_type=request.case_type,
            query=query,
            top_houses=top,
            debug={"response_kind": "search", "plan": plan.plan_type, "candidate_count": len(candidates)},
        )

    async def _init_session_data(self, state: SessionState) -> None:
        try:
            log_event(LOGGER, "dialogue.session.init.start")
            await self.houses_client.init_houses()
            self.cache.invalidate_all_houses()
            log_event(LOGGER, "dialogue.session.init.done")
        except DataSourceError:
            LOGGER.warning("init houses failed for session=%s", state.session_id)
            log_event(LOGGER, "dialogue.session.init.failed")

    async def _handle_house_detail(self, query: StructuredQuery, state: SessionState) -> InvokeResponse:
        house_id = query.hard.house_id
        if not house_id:
            return InvokeResponse(
                text="请告诉我要看的房源（例如 HF_2001，或说第一套/这套）。",
                debug={"response_kind": "detail"},
            )

        house = await self._load_house_detail(house_id)
        if house is None:
            return InvokeResponse(
                text=f"未找到房源 {house_id} 的详情，请确认 house_id 是否正确。",
                debug={"response_kind": "detail", "referenced_house_ids": [house_id]},
            )

        listings = await self._load_listings(house_id)
        preferred_platform = self._choose_preferred_platform(listings) or self._platform_from_text(house.listing_platform)
        state.focus_house_id = house_id
        if preferred_platform is not None:
            state.focus_listing_platform = preferred_platform

        details: list[str] = []
        location = "".join([house.district or "", house.community or ""])
        headline = f"{house_id}："
        if location:
            headline += f"{location}，"
        if house.layout:
            headline += f"{house.layout}，"
        if house.area is not None:
            headline += f"{house.area:.1f}㎡，"
        if house.rent is not None:
            headline += f"{house.rent} 元/月。"
        else:
            headline += "价格待确认。"
        details.append(headline)
        if house.subway_distance is not None:
            details.append(f"离最近地铁约 {house.subway_distance} 米。")
        if house.commute_to_xierqi_min is not None:
            details.append(f"到西二旗通勤约 {house.commute_to_xierqi_min} 分钟。")
        if house.status:
            details.append(f"当前状态：{house.status}。")

        listing_summary = self._summarize_listings(listings)
        if listing_summary:
            details.append(listing_summary)

        return InvokeResponse(
            text="".join(details),
            debug={"response_kind": "detail", "referenced_house_ids": [house_id]},
        )

    async def _handle_listings_query(self, query: StructuredQuery, state: SessionState) -> InvokeResponse:
        house_id = query.hard.house_id
        if not house_id:
            return InvokeResponse(
                text="请先告诉我房源 house_id（例如 HF_2001），或者说“第一套”。",
                debug={"response_kind": "detail"},
            )

        listings = await self._load_listings(house_id)
        if not listings:
            state.focus_house_id = house_id
            return InvokeResponse(
                text=f"{house_id} 暂未查到平台挂牌记录。",
                debug={"response_kind": "detail", "referenced_house_ids": [house_id]},
            )

        state.focus_house_id = house_id
        preferred = self._choose_preferred_platform(listings)
        if preferred is not None:
            state.focus_listing_platform = preferred

        order = ["安居客", "链家", "58同城"]
        listing_map = {item.listing_platform: item for item in listings if item.listing_platform}
        rendered: list[str] = []
        for platform in order:
            item = listing_map.get(platform)
            if not item:
                continue
            rent_text = f"{item.rent} 元" if item.rent is not None else "价格缺失"
            status_text = item.status or "状态未知"
            rendered.append(f"{platform} {rent_text}（{status_text}）")
        if not rendered:
            rendered = [
                f"{item.listing_platform or '未知平台'} {item.rent if item.rent is not None else '价格缺失'}（{item.status or '状态未知'}）"
                for item in listings
            ]

        text = f"{house_id} 各平台挂牌：{'；'.join(rendered)}。"
        if preferred is not None:
            text += f" 如需办理租房，优先 {preferred.value}。"
        return InvokeResponse(
            text=text,
            debug={"response_kind": "detail", "referenced_house_ids": [house_id]},
        )

    async def _handle_action(self, query: StructuredQuery, state: SessionState) -> InvokeResponse:
        house_id = query.hard.house_id or state.focus_house_id
        if not house_id:
            return InvokeResponse(
                text="请提供要操作的房源 house_id（例如 HF_2001），或先说“租第一套”。",
                debug={"response_kind": "action"},
            )

        platform = query.hard.listing_platform or state.focus_listing_platform
        if platform is None:
            platform = self._choose_preferred_platform(await self._load_listings(house_id))
        if platform is None:
            platform = Platform.anjuke

        state.phase = SessionPhase.executing
        log_event(
            LOGGER,
            "dialogue.action.start",
            intent=query.intent.value,
            house_id=house_id,
            listing_platform=platform.value,
        )

        if query.intent == IntentType.rent:
            result = await self.houses_client.rent(house_id, platform.value)
            action = "rent"
            message = "已提交租房操作"
        elif query.intent == IntentType.terminate:
            result = await self.houses_client.terminate(house_id, platform.value)
            action = "terminate"
            message = "已提交退租操作"
        else:
            result = await self.houses_client.offline(house_id, platform.value)
            action = "offline"
            message = "已提交下架操作"

        self.cache.invalidate_house(house_id)
        self.cache.invalidate_query_cache()
        state.phase = SessionPhase.presenting
        state.focus_house_id = house_id
        state.focus_listing_platform = platform
        log_event(LOGGER, "dialogue.action.done", action=action, result=result)
        return InvokeResponse(
            text=f"{message}：{house_id}（{platform.value}）。",
            debug={"response_kind": "action", "action_result": result, "referenced_house_ids": [house_id]},
        )

    async def _load_house_detail(self, house_id: str) -> HouseLite | None:
        cached = self.cache.house_detail.get(house_id)
        if cached is not None:
            return cached
        detail = await self.houses_client.get_house_detail(house_id)
        if detail is not None:
            self.cache.house_detail[house_id] = detail
        return detail

    async def _load_listings(self, house_id: str) -> list[Listing]:
        cached = self.cache.house_listings.get(house_id)
        if cached is not None:
            return cached
        page = await self.houses_client.get_listings(house_id)
        items = page.get("items", [])
        listings = [item for item in items if isinstance(item, Listing)]
        self.cache.house_listings[house_id] = listings
        return listings

    @staticmethod
    def _has_actionable_preferences(query: StructuredQuery) -> bool:
        hard = query.hard
        soft = query.soft
        return any(
            [
                hard.layout,
                hard.area_min is not None,
                hard.max_subway_dist is not None,
                hard.max_distance is not None,
                hard.max_commute_min is not None,
                hard.rent_type,
                hard.utilities_type,
                soft.orientation,
                soft.decoration,
                soft.elevator is not None,
                soft.noise_preference,
                bool(soft.amenities),
            ]
        )

    @staticmethod
    def _remember_chat_preferences(state: SessionState, query: StructuredQuery) -> None:
        hard = HardConstraints.model_validate(state.confirmed_constraints.model_dump())
        soft = SoftPreferences.model_validate(state.soft_preferences.model_dump())

        for key, value in query.hard.model_dump().items():
            if value is not None:
                setattr(hard, key, value)

        for key, value in query.soft.model_dump().items():
            if isinstance(value, list) and value:
                merged = sorted(set(getattr(soft, key) + value))
                setattr(soft, key, merged)
            elif value is not None:
                setattr(soft, key, value)

        state.confirmed_constraints = hard
        state.soft_preferences = soft

    @staticmethod
    def _extract_llm_assistant_reply(meta: dict[str, Any] | None) -> str | None:
        if not isinstance(meta, dict):
            return None
        llm_parse = meta.get("llm_parse")
        if not isinstance(llm_parse, dict):
            return None
        assistant_reply = llm_parse.get("assistant_reply")
        if isinstance(assistant_reply, str):
            stripped = assistant_reply.strip()
            if stripped:
                return stripped
        return None

    @staticmethod
    def _fallback_chat_reply(user_text: str, query: StructuredQuery) -> str:
        prefs: list[str] = []
        if query.soft.orientation:
            prefs.append(f"朝向{query.soft.orientation}")
        if query.hard.area_min is not None:
            prefs.append(f"面积至少{int(query.hard.area_min)}平")
        if query.hard.max_subway_dist is not None:
            prefs.append(f"地铁距离不超过{query.hard.max_subway_dist}米")
        if query.hard.max_commute_min is not None:
            prefs.append(f"通勤不超过{query.hard.max_commute_min}分钟")

        if prefs:
            return f"已记录你提到的偏好：{'，'.join(prefs)}。"
        cleaned = preview_text(user_text, limit=60).strip()
        if cleaned:
            return f"已收到你的反馈：{cleaned}"
        return "已收到你的反馈。"

    def _merge_query_with_state(self, query: StructuredQuery, state: SessionState, user_text: str) -> StructuredQuery:
        hard = HardConstraints.model_validate(state.confirmed_constraints.model_dump())
        soft = SoftPreferences.model_validate(state.soft_preferences.model_dump())

        replace_location = query.intent == IntentType.search and any(sig in user_text for sig in _REPLACE_SIGNALS) and any(
            [query.hard.district, query.hard.area, query.hard.community, query.hard.landmark_name]
        )
        if replace_location:
            hard.district = None
            hard.area = None
            hard.community = None
            hard.landmark_name = None
            hard.landmark_category = None

        for field_name, value in query.hard.model_dump().items():
            if value is not None:
                setattr(hard, field_name, value)

        if query.hard.community is not None:
            hard.landmark_name = None
            hard.landmark_category = None
        if query.hard.landmark_name is not None:
            hard.community = None
        if replace_location and query.hard.district is not None and query.hard.landmark_name is None:
            hard.landmark_name = None
            hard.landmark_category = None
            hard.area = query.hard.area

        for field_name, value in query.soft.model_dump().items():
            if isinstance(value, list) and value:
                merged = sorted(set(getattr(soft, field_name) + value))
                setattr(soft, field_name, merged)
            elif value is not None:
                setattr(soft, field_name, value)

        query.hard = hard
        query.soft = soft
        return query

    def _apply_llm_parse(self, query: StructuredQuery, llm_parse: Any) -> StructuredQuery:
        if not isinstance(llm_parse, dict):
            return query

        confidence = _to_float(llm_parse.get("confidence"))
        if confidence is not None:
            query.confidence = confidence

        parsed_intent = _normalize_intent(llm_parse.get("intent"))
        if parsed_intent is not None:
            query.intent = parsed_intent

        tool_plan = llm_parse.get("tool_plan")
        if isinstance(tool_plan, dict):
            op = tool_plan.get("operationId")
            mapped_intent = _map_tool_operation_to_intent(op)
            if mapped_intent is not None:
                query.intent = mapped_intent
            args = tool_plan.get("arguments")
            if isinstance(args, dict):
                normalized_args = self._normalize_tool_arguments(args)
                self._apply_hard_overrides(query.hard, normalized_args)
                self._apply_soft_overrides(query.soft, normalized_args)

        hard_raw = llm_parse.get("hard")
        if isinstance(hard_raw, dict):
            self._apply_hard_overrides(query.hard, hard_raw)

        soft_raw = llm_parse.get("soft")
        if isinstance(soft_raw, dict):
            self._apply_soft_overrides(query.soft, soft_raw)

        return query

    def _normalize_tool_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in args.items():
            if key in _TOOL_ARGUMENT_ALIASES:
                normalized[_TOOL_ARGUMENT_ALIASES[key]] = value
            else:
                normalized[key] = value

        bedrooms = normalized.get("bedrooms")
        if bedrooms is not None and "layout" not in normalized:
            parsed = _to_int(bedrooms)
            if parsed is not None and parsed > 0:
                normalized["layout"] = f"{parsed}居"

        category = normalized.get("type")
        if isinstance(category, str):
            category_lower = category.strip().lower()
            if category_lower == "shopping":
                normalized["amenities"] = ["商超"]
            elif category_lower == "park":
                normalized["amenities"] = ["公园"]
        return normalized

    def _apply_hard_overrides(self, hard: HardConstraints, payload: dict[str, Any]) -> None:
        for key in HardConstraints.model_fields:
            if key not in payload:
                continue
            value = payload.get(key)
            if value is None:
                continue

            if key == "listing_platform":
                platform = self._platform_from_text(str(value))
                if platform is not None:
                    hard.listing_platform = platform
                continue
            if key in {"budget_min", "budget_max", "max_subway_dist", "max_distance", "max_commute_min"}:
                parsed = _to_int(value)
                if parsed is not None:
                    setattr(hard, key, parsed)
                continue
            if key == "area_min":
                parsed_float = _to_float(value)
                if parsed_float is not None:
                    hard.area_min = parsed_float
                continue
            if key == "house_id":
                if isinstance(value, str) and value.strip():
                    hard.house_id = value.strip().upper()
                continue

            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    setattr(hard, key, cleaned)

    @staticmethod
    def _apply_soft_overrides(soft: SoftPreferences, payload: dict[str, Any]) -> None:
        for key in SoftPreferences.model_fields:
            if key not in payload:
                continue
            value = payload.get(key)
            if value is None:
                continue

            if key == "amenities":
                if isinstance(value, list):
                    tokens = [str(item).strip() for item in value if str(item).strip()]
                elif isinstance(value, str):
                    tokens = [item.strip() for item in re.split(r"[,，/、\s]+", value) if item.strip()]
                else:
                    tokens = []
                if tokens:
                    soft.amenities = sorted(set(soft.amenities + tokens))
                continue

            if key in {"elevator", "value_for_money", "prioritize_subway_distance"}:
                parsed_bool = _to_bool(value)
                if parsed_bool is not None:
                    setattr(soft, key, parsed_bool)
                continue

            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    setattr(soft, key, cleaned)

    def _resolve_house_id(self, user_text: str, query: StructuredQuery, state: SessionState) -> str | None:
        if query.hard.house_id:
            return query.hard.house_id

        idx = self._extract_rank_index(user_text)
        if idx is not None:
            latest_ids = self._latest_house_ids(state)
            if 1 <= idx <= len(latest_ids):
                return latest_ids[idx - 1]

        if "最开始" in user_text or "最初" in user_text:
            for snapshot in state.search_history:
                if snapshot.landmark_name and snapshot.landmark_name in user_text and snapshot.house_ids:
                    return snapshot.house_ids[0]
            if state.search_history and state.search_history[0].house_ids:
                return state.search_history[0].house_ids[0]

        for snapshot in state.search_history:
            if snapshot.landmark_name and snapshot.landmark_name in user_text and snapshot.house_ids:
                return snapshot.house_ids[0]
            if snapshot.community and snapshot.community in user_text and snapshot.house_ids:
                return snapshot.house_ids[0]
            if snapshot.district and snapshot.district in user_text and snapshot.house_ids:
                return snapshot.house_ids[0]

        if any(word in user_text for word in _HOUSE_REF_WORDS):
            if state.focus_house_id:
                return state.focus_house_id
            latest_ids = self._latest_house_ids(state)
            if latest_ids:
                return latest_ids[0]
        return None

    @staticmethod
    def _extract_explicit_house_id(text: str) -> str | None:
        match = _EXPLICIT_HOUSE_ID_PATTERN.search(text)
        if not match:
            return None
        return match.group(1).upper()

    @staticmethod
    def _extract_rank_index(text: str) -> int | None:
        match = _RANK_INDEX_PATTERN.search(text)
        if not match:
            return None
        raw = match.group(1)
        if raw.isdigit():
            return int(raw)
        return _CH_NUM_TO_INT.get(raw)

    @staticmethod
    def _platform_from_text(value: str | None) -> Platform | None:
        if not value:
            return None
        if "58" in value:
            return Platform.wuba
        if value == Platform.lianjia.value:
            return Platform.lianjia
        if value == Platform.anjuke.value:
            return Platform.anjuke
        return None

    @staticmethod
    def _choose_preferred_platform(listings: list[Listing]) -> Platform | None:
        if not listings:
            return None
        rentable = [row for row in listings if row.listing_platform and row.status and _is_rentable_status(row.status)]
        pool = rentable or [row for row in listings if row.listing_platform]
        if not pool:
            return None
        best = sorted(pool, key=lambda row: row.rent if row.rent is not None else 10**9)[0]
        return DialogueManager._platform_from_text(best.listing_platform)

    @staticmethod
    def _summarize_listings(listings: list[Listing]) -> str:
        if not listings:
            return ""
        order = ["安居客", "链家", "58同城"]
        mapping = {item.listing_platform: item for item in listings if item.listing_platform}
        chunks: list[str] = []
        for platform in order:
            item = mapping.get(platform)
            if not item:
                continue
            rent_text = f"{item.rent} 元" if item.rent is not None else "价格缺失"
            status_text = item.status or "状态未知"
            chunks.append(f"{platform} {rent_text}（{status_text}）")
        if not chunks:
            return ""
        return "挂牌情况：" + "；".join(chunks) + "。"

    @staticmethod
    def _compress_candidates(candidates: list[HouseLite], keep: int = 40) -> list[HouseLite]:
        compact: list[HouseLite] = []
        for house in candidates[:keep]:
            compact.append(
                HouseLite(
                    house_id=house.house_id,
                    rent=house.rent,
                    layout=house.layout,
                    area=house.area,
                    district=house.district,
                    community=house.community,
                    subway_distance=house.subway_distance,
                    commute_to_xierqi_min=house.commute_to_xierqi_min,
                    status=house.status,
                    tags=house.tags,
                    decoration=house.decoration,
                    elevator=house.elevator,
                    orientation=house.orientation,
                    available_date=house.available_date,
                    listing_platform=house.listing_platform,
                )
            )
        return compact

    @staticmethod
    def _latest_house_ids(state: SessionState) -> list[str]:
        if state.search_history and state.search_history[-1].house_ids:
            return state.search_history[-1].house_ids
        return [item.house_id for item in state.last_top5]

    def _remember_search_snapshot(self, state: SessionState, user_text: str, query: StructuredQuery) -> None:
        house_ids = [item.house_id for item in state.last_top5]
        if house_ids:
            snapshot = SearchSnapshot(
                query_text=preview_text(user_text, limit=100),
                district=query.hard.district,
                area=query.hard.area,
                community=query.hard.community,
                landmark_name=query.hard.landmark_name,
                house_ids=house_ids,
            )
            state.search_history.append(snapshot)
            state.search_history = state.search_history[-8:]
            state.focus_house_id = house_ids[0]
            platform = self._platform_from_text(state.last_top5[0].listing_platform if state.last_top5 else None)
            if platform is not None:
                state.focus_listing_platform = platform

    def _remember_turn(self, state: SessionState, user_text: str, response: InvokeResponse, intent: IntentType) -> None:
        house_ids = [item.house_id for item in response.candidates]
        for ref_house_id in response.debug.get("referenced_house_ids", []):
            if isinstance(ref_house_id, str) and ref_house_id not in house_ids:
                house_ids.append(ref_house_id)

        state.recent_turns.append(
            TurnSummary(
                user=preview_text(user_text, limit=120),
                assistant=preview_text(response.text, limit=180),
                intent=intent,
                house_ids=house_ids,
            )
        )
        state.recent_turns = state.recent_turns[-12:]
        state.conversation_summary = self._build_conversation_summary(state)

    @staticmethod
    def _build_conversation_summary(state: SessionState) -> str:
        constraints: list[str] = []
        if state.confirmed_constraints.district:
            constraints.append(f"区域={state.confirmed_constraints.district}")
        if state.confirmed_constraints.landmark_name:
            constraints.append(f"地标={state.confirmed_constraints.landmark_name}")
        if state.confirmed_constraints.community:
            constraints.append(f"小区={state.confirmed_constraints.community}")
        if state.confirmed_constraints.layout:
            constraints.append(f"户型={state.confirmed_constraints.layout}")
        if state.confirmed_constraints.budget_max is not None:
            constraints.append(f"预算上限={state.confirmed_constraints.budget_max}")

        focus_parts: list[str] = []
        if state.focus_house_id:
            focus_parts.append(f"house={state.focus_house_id}")
        if state.focus_listing_platform:
            focus_parts.append(f"platform={state.focus_listing_platform.value}")

        search_notes: list[str] = []
        for idx, snapshot in enumerate(state.search_history[-4:]):
            if not snapshot.house_ids:
                continue
            anchor = snapshot.landmark_name or snapshot.community or snapshot.district or "-"
            search_notes.append(f"{idx + 1}:{anchor}:{'/'.join(snapshot.house_ids[:2])}")

        action_notes = [
            f"{turn.intent.value}:{'/'.join(turn.house_ids[:2])}"
            for turn in state.recent_turns[-6:]
            if turn.intent in {IntentType.rent, IntentType.terminate, IntentType.offline} and turn.house_ids
        ]

        blocks: list[str] = []
        if constraints:
            blocks.append("约束[" + "，".join(constraints) + "]")
        if focus_parts:
            blocks.append("焦点[" + "，".join(focus_parts) + "]")
        if search_notes:
            blocks.append("检索[" + "；".join(search_notes) + "]")
        if action_notes:
            blocks.append("动作[" + "；".join(action_notes) + "]")

        summary = " ".join(blocks)
        return summary[:700]


def _is_rentable_status(status: str) -> bool:
    normalized = status.strip().lower()
    return normalized in {"可租", "available"}


def _normalize_intent(value: Any) -> IntentType | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    for item in IntentType:
        if normalized == item.value:
            return item
    return None


def _map_tool_operation_to_intent(value: Any) -> IntentType | None:
    if not isinstance(value, str):
        return None
    return _TOOL_OPERATION_TO_INTENT.get(value.strip())


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "是"}:
            return True
        if normalized in {"false", "0", "no", "n", "否"}:
            return False
    return None
