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
from app.clients.landmarks import LandmarksClient
from app.infra.cache import CacheManager
from app.infra.logging import log_event, preview_text
from app.infra.tool_recorder import record_tool_result
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
    "通勤时间太长",
    "噪音大",
)
_SEARCH_SIGNALS = ("找房", "找个房", "推荐", "筛选", "查询", "搜", "帮我找", "看看房", "有房", "房源")
_RENT_STATUS_QUERY_SIGNALS = ("可租吗", "可以租吗", "能租吗", "能不能租", "是否可租", "能否租")
_RENT_COMMIT_SIGNALS = ("我要租", "我想租", "帮我租", "租这套", "租这一套", "租这个", "就租", "办理租房")
_GENERIC_SEARCH_START_SIGNALS = (
    "想换个房子",
    "想换房子",
    "想找房子",
    "帮我找找",
    "能帮我找",
    "可以帮我找",
    "帮我找房子",
)
_ADMIN_DIVISION_SUFFIXES = ("区", "县", "旗", "市", "州", "盟")
_CONTEXT_CONTINUE_SIGNALS = ("按之前", "按刚才", "按上次", "按这些", "就按这个", "继续找", "继续推荐", "照这个条件")
_DIRECT_REQUIREMENT_SIGNALS = (
    "朝南",
    "采光好",
    "近地铁",
    "离地铁近",
    "通勤方便",
    "通勤短",
    "合租",
    "整租",
    "民水民电",
    "商水商电",
    "有电梯",
    "安静",
    "公园",
    "商场",
)
_STICKY_SOFT_BOOL_FIELDS = {"prefer_spacious", "prioritize_subway_distance", "prioritize_commute"}
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
        landmarks_client: LandmarksClient | None = None,
        cache: CacheManager,
        max_output_candidates: int,
    ) -> None:
        self.state_store = state_store
        self.nlu = nlu
        self.planner = planner
        self.ranker = ranker
        self.formatter = formatter
        self.houses_client = houses_client
        self.landmarks_client = landmarks_client
        self.cache = cache
        self.max_output_candidates = max_output_candidates
        self._known_district_aliases: set[str] | None = None

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
        if merged.intent == IntentType.search:
            await self._normalize_search_location_slots(merged, request.message)
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
        if merged.intent == IntentType.rent and self._is_rent_status_query(request.message):
            log_event(
                LOGGER,
                "dialogue.intent.adjusted",
                original_intent=IntentType.rent.value,
                adjusted_intent=IntentType.house_detail.value,
                reason="rent_status_question_without_commitment",
            )
            merged.intent = IntentType.house_detail
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
                resp = await self._handle_house_detail(merged, state, request.message)
            elif merged.intent == IntentType.compare:
                resp = self._handle_compare(merged, state, request.message)
            elif merged.intent == IntentType.search and state.phase in {
                SessionPhase.chatting,
                SessionPhase.slot_filling,
            }:
                follow_up = self._build_search_follow_up_questions(merged, request.message)
                if follow_up:
                    state.phase = SessionPhase.slot_filling
                    assistant_reply = self._extract_llm_assistant_reply(request.meta)
                    resp = InvokeResponse(
                        text=assistant_reply or self._build_follow_up_text(follow_up),
                        clarify_questions=follow_up,
                        debug={"response_kind": "clarify"},
                    )
                else:
                    resp = await self._handle_search(merged, request, state)
            elif merged.intent == IntentType.chat:
                log_event(LOGGER, "dialogue.chat")
                self._remember_chat_preferences(state, merged, request.message)
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
        if any(token in text for token in _COMPLAINT_SIGNALS):
            return True
        if re.search(r"通勤.{0,4}太长", text):
            return True
        return False

    def _should_treat_search_as_chat(self, text: str, query: StructuredQuery) -> bool:
        if query.intent != IntentType.search:
            return False
        if self._has_explicit_search_request(text):
            return False
        if not self._has_complaint_signal(text):
            return False
        return self._has_actionable_preferences(query) or self._text_implies_preferences(text)

    @staticmethod
    def _text_implies_preferences(text: str) -> bool:
        inferred_hard = HardConstraints()
        inferred_soft = SoftPreferences()
        DialogueManager._infer_preferences_from_user_text(text, inferred_hard, inferred_soft)
        inferred_query = StructuredQuery(hard=inferred_hard, soft=inferred_soft)
        return DialogueManager._has_actionable_preferences(inferred_query)

    @staticmethod
    def _is_rent_status_query(text: str) -> bool:
        asks_status = any(token in text for token in _RENT_STATUS_QUERY_SIGNALS)
        commits_rent = any(token in text for token in _RENT_COMMIT_SIGNALS)
        return asks_status and not commits_rent

    @staticmethod
    def _build_search_follow_up_questions(query: StructuredQuery, user_text: str) -> list[str]:
        hard = query.hard
        current_has_constraints = DialogueManager._text_has_explicit_search_constraints(user_text) or DialogueManager._text_has_direct_house_requirements(
            user_text
        )
        context_continue = DialogueManager._text_requests_context_continuation(user_text)
        has_location = any([hard.district, hard.area, hard.community, hard.landmark_name, hard.landmark_id])
        has_budget = hard.budget_max is not None or hard.budget_min is not None
        has_actionable_context = DialogueManager._has_actionable_preferences(query)
        has_searchable_signal = has_location or has_budget or has_actionable_context

        if not current_has_constraints and not context_continue and not has_searchable_signal:
            return [
                "你的预算上限大概是多少（元/月）？",
                "你更偏向哪个区域、小区或地铁站附近？",
            ]

        if context_continue and has_actionable_context:
            return []

        if has_location or has_budget:
            return []

        if current_has_constraints and has_actionable_context:
            return []

        questions: list[str] = []
        questions.append("你的预算上限大概是多少（元/月）？")
        questions.append("你更偏向哪个区域、小区或地铁站附近？")
        return questions

    @staticmethod
    def _build_follow_up_text(questions: list[str]) -> str:
        if not questions:
            return "请补充你的找房偏好。"
        lines = [f"{idx + 1}. {q}" for idx, q in enumerate(questions)]
        return "为了给你推荐更匹配的房源，请先补充这两点：\n" + "\n".join(lines)

    def _handle_compare(self, query: StructuredQuery, state: SessionState, user_text: str) -> InvokeResponse:
        _ = query
        if not state.last_top5:
            return InvokeResponse(
                text="我可以帮你做多房源对比。先告诉我找房条件，我先筛出候选后再给你比价和决策建议。",
                clarify_questions=["先说下你的预算上限和目标区域。"],
                debug={"response_kind": "clarify"},
            )

        requested_ids = self._extract_compare_house_ids(user_text)
        selected = state.last_top5[:5]
        if requested_ids:
            picked = [row for row in state.last_top5 if row.house_id in requested_ids]
            if picked:
                selected = picked[:5]
        if len(selected) < 2 and len(state.last_top5) >= 2:
            selected = state.last_top5[: min(5, len(state.last_top5))]

        if not selected:
            return InvokeResponse(
                text="当前没有可对比的候选房源，请先让我帮你筛选房源。",
                clarify_questions=["你可以先说：预算、区域、户型。"],
                debug={"response_kind": "clarify"},
            )

        recommend = selected[0]
        state.focus_house_id = recommend.house_id
        preferred = self._platform_from_text(recommend.listing_platform)
        if preferred is not None:
            state.focus_listing_platform = preferred

        lines: list[str] = ["已根据当前候选做多维对比（租金/通勤/地铁/户型）："]
        for idx, row in enumerate(selected, start=1):
            parts: list[str] = []
            if row.rent is not None:
                parts.append(f"租金{row.rent}元")
            if row.commute_to_xierqi_min is not None:
                parts.append(f"通勤{row.commute_to_xierqi_min}分钟")
            if row.subway_distance is not None:
                parts.append(f"地铁{row.subway_distance}米")
            if row.layout:
                parts.append(row.layout)
            if row.area is not None:
                parts.append(f"{row.area:.0f}㎡")
            line = f"{idx}. {row.house_id}"
            if parts:
                line += f"（{'，'.join(parts)}）"
            lines.append(line)
        lines.append(f"当前综合优先推荐：{recommend.house_id}。如果你说“这套不错，我要租这套”，我会继续办理。")
        return InvokeResponse(
            text="\n".join(lines),
            candidates=selected,
            debug={"response_kind": "compare", "referenced_house_ids": [row.house_id for row in selected]},
        )

    @staticmethod
    def _extract_compare_house_ids(text: str) -> list[str]:
        if not isinstance(text, str) or not text:
            return []
        raw_ids = _EXPLICIT_HOUSE_ID_PATTERN.findall(text)
        if not raw_ids:
            return []
        unique: list[str] = []
        seen: set[str] = set()
        for item in raw_ids:
            normalized = item.upper()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique

    @staticmethod
    def _is_generic_search_start(text: str) -> bool:
        return any(token in text for token in _GENERIC_SEARCH_START_SIGNALS)

    @staticmethod
    def _text_has_explicit_search_constraints(text: str) -> bool:
        if DialogueManager._contains_admin_division_token(text):
            return True
        if any(token in text for token in ("小区", "地铁站", "附近", "商圈", "区域")):
            return True
        if re.search(r"\d+\s*(元|块|w|万|k|千|米|分钟|分)", text):
            return True
        if re.search(r"[一二两三四五六七八九1-9]\s*(居|室)", text):
            return True
        return False

    @staticmethod
    def _text_has_direct_house_requirements(text: str) -> bool:
        if any(token in text for token in _DIRECT_REQUIREMENT_SIGNALS):
            return True
        if re.search(r"(大一点|面积大|房间大)", text):
            return True
        return False

    @staticmethod
    def _text_requests_context_continuation(text: str) -> bool:
        if any(token in text for token in _CONTEXT_CONTINUE_SIGNALS):
            return True
        if re.search(r"按之前.*条件", text):
            return True
        if re.search(r"(继续|接着).*(找房|推荐|筛选)", text):
            return True
        return False

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

    async def _handle_house_detail(self, query: StructuredQuery, state: SessionState, user_text: str) -> InvokeResponse:
        if self._should_summarize_top5_subway_distance(user_text, query, state):
            return await self._summarize_top5_subway_distance(state)

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

    async def _summarize_top5_subway_distance(self, state: SessionState) -> InvokeResponse:
        selected = state.last_top5[:5]
        rows: list[dict[str, Any]] = []
        rendered: list[str] = []
        referenced: list[str] = []

        for idx, item in enumerate(selected, start=1):
            house_id = item.house_id
            referenced.append(house_id)

            detail = await self._load_house_detail(house_id)
            distance = detail.subway_distance if detail and detail.subway_distance is not None else item.subway_distance
            nearest_subway = getattr(item, "nearest_subway", None)
            if detail and getattr(detail, "subway_distance", None) is None:
                nearest_subway = None

            rows.append(
                {
                    "house_id": house_id,
                    "subway_distance": distance,
                    "nearest_subway": nearest_subway,
                }
            )
            if distance is None:
                rendered.append(f"{idx}. {house_id}：地铁距离信息暂未提供")
            else:
                rendered.append(f"{idx}. {house_id}：离地铁约 {distance} 米")

        if referenced:
            state.focus_house_id = referenced[0]

        record_tool_result(
            name="SESSION_TOP5_SUBWAY_DISTANCE",
            success=True,
            output={"items": rows},
            duration_ms=0,
            method="LOCAL",
            url="memory://session/last_top5",
            status_code=None,
        )

        return InvokeResponse(
            text="；".join(rendered) + "。",
            debug={"response_kind": "detail", "referenced_house_ids": referenced, "detail_mode": "top5_subway_distance"},
        )

    @staticmethod
    def _should_summarize_top5_subway_distance(text: str, query: StructuredQuery, state: SessionState) -> bool:
        if not state.last_top5:
            return False
        if not isinstance(text, str) or not text:
            return False
        if not any(token in text for token in ("地铁", "站")):
            return False
        if not any(token in text for token in ("多远", "距离", "多少米", "几米")):
            return False
        if _EXPLICIT_HOUSE_ID_PATTERN.search(text):
            return False
        if query.hard.house_id and "这套" not in text and "这一套" not in text:
            return False
        return True

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

        requested_platform = query.hard.listing_platform
        listings = await self._load_listings(house_id)
        platform = requested_platform or state.focus_listing_platform
        if platform is None:
            platform = self._choose_preferred_platform(listings)
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
            selected_platform, listing_status = self._resolve_rent_platform_and_status(
                requested_platform=requested_platform,
                current_platform=platform,
                listings=listings,
            )
            if selected_platform is not None:
                platform = selected_platform
            detail = await self._load_house_detail(house_id)
            detail_status = detail.status if detail else None
            effective_status = detail_status or listing_status
            if not effective_status or not _is_rentable_status(effective_status):
                state.phase = SessionPhase.presenting
                blocked_status = effective_status or "状态未知"
                return InvokeResponse(
                    text=f"{house_id} 当前状态为「{blocked_status}」，暂不可直接办理租房。",
                    debug={"response_kind": "detail", "referenced_house_ids": [house_id]},
                )
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

    def _resolve_rent_platform_and_status(
        self,
        *,
        requested_platform: Platform | None,
        current_platform: Platform,
        listings: list[Listing],
    ) -> tuple[Platform | None, str | None]:
        if not listings:
            return current_platform, None

        if requested_platform is not None:
            requested_listing = self._find_listing_for_platform(listings, requested_platform)
            if requested_listing and requested_listing.status:
                return requested_platform, requested_listing.status
            return requested_platform, None

        current_listing = self._find_listing_for_platform(listings, current_platform)
        if current_listing and current_listing.status and _is_rentable_status(current_listing.status):
            return current_platform, current_listing.status

        for item in listings:
            if not item.status or not _is_rentable_status(item.status):
                continue
            platform = self._platform_from_text(item.listing_platform)
            if platform is not None:
                return platform, item.status

        if current_listing and current_listing.status:
            return current_platform, current_listing.status
        for item in listings:
            if item.status:
                return current_platform, item.status
        return current_platform, None

    def _find_listing_for_platform(self, listings: list[Listing], platform: Platform) -> Listing | None:
        for item in listings:
            normalized = self._platform_from_text(item.listing_platform)
            if normalized == platform:
                return item
        return None

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
                soft.prefer_spacious,
                soft.prioritize_subway_distance,
                soft.prioritize_commute,
            ]
        )

    @staticmethod
    def _remember_chat_preferences(state: SessionState, query: StructuredQuery, user_text: str) -> None:
        hard = HardConstraints.model_validate(state.confirmed_constraints.model_dump())
        soft = SoftPreferences.model_validate(state.soft_preferences.model_dump())

        for key, value in query.hard.model_dump().items():
            if value is not None:
                setattr(hard, key, value)

        for key, value in query.soft.model_dump().items():
            if isinstance(value, list) and value:
                merged = sorted(set(getattr(soft, key) + value))
                setattr(soft, key, merged)
            elif isinstance(value, bool) and key in _STICKY_SOFT_BOOL_FIELDS:
                if value:
                    setattr(soft, key, True)
            elif value is not None:
                setattr(soft, key, value)

        if DialogueManager._has_complaint_signal(user_text) and not DialogueManager._has_explicit_search_request(user_text):
            # Complaint turns should be remembered as ranking preferences, not concrete hard filters.
            hard.area_min = None
            hard.max_subway_dist = None
            hard.max_commute_min = None

        DialogueManager._infer_preferences_from_user_text(user_text, hard, soft)
        state.confirmed_constraints = hard
        state.soft_preferences = soft

    @staticmethod
    def _infer_preferences_from_user_text(user_text: str, hard: HardConstraints, soft: SoftPreferences) -> None:
        text = user_text or ""
        if soft.orientation is None and any(token in text for token in ("采光不好", "阴暗", "不朝阳")):
            soft.orientation = "朝南"
        small_room = any(token in text for token in ("房间小", "太小", "住的不舒服", "不太舒服")) or bool(re.search(r"房间.{0,2}小", text))
        if small_room:
            soft.prefer_spacious = True
        long_commute = any(token in text for token in ("通勤时间长", "通勤太长", "每天都要早起", "上班太远")) or bool(
            re.search(r"通勤.{0,4}(太长|很长|过长|太远|很远|远)", text)
        )
        if long_commute:
            soft.prioritize_subway_distance = True
            soft.prioritize_commute = True

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
        if query.soft.prefer_spacious:
            prefs.append("面积更大优先")
        if query.soft.prioritize_subway_distance:
            prefs.append("地铁距离更近优先")
        if query.soft.prioritize_commute:
            prefs.append("通勤更短优先")
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
            elif isinstance(value, bool) and field_name in _STICKY_SOFT_BOOL_FIELDS:
                if value:
                    setattr(soft, field_name, True)
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
            if key == "district":
                if isinstance(value, str):
                    normalized_district, normalized_area = self._normalize_district_or_area(value)
                    if normalized_district is not None:
                        hard.district = normalized_district
                    if normalized_area is not None:
                        hard.area = normalized_area
                        if normalized_district is None:
                            hard.district = None
                continue
            if key == "area":
                if isinstance(value, str):
                    cleaned_area = value.strip()
                    if cleaned_area:
                        hard.area = cleaned_area
                continue

            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    setattr(hard, key, cleaned)

    @staticmethod
    def _normalize_district_or_area(value: str) -> tuple[str | None, str | None]:
        cleaned = value.strip()
        if not cleaned:
            return None, None

        if DialogueManager._looks_like_admin_division(cleaned):
            normalized = cleaned[:-1] if cleaned.endswith("区") else cleaned
            return (normalized or cleaned), None
        # Keep ambiguous values in district first; async normalization will refine using
        # user text and upstream landmark districts to avoid city-specific hardcoding.
        return cleaned, None

    @staticmethod
    def _looks_like_admin_division(value: str) -> bool:
        cleaned = value.strip()
        if not cleaned:
            return False
        if cleaned.endswith(("新区", "开发区", "自治县", "自治州")):
            return True
        return cleaned.endswith(_ADMIN_DIVISION_SUFFIXES)

    @staticmethod
    def _contains_admin_division_token(text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"[\u4e00-\u9fa5A-Za-z0-9]{1,20}(新区|开发区|自治县|自治州|区|县|旗|市|州|盟)", text))

    async def _normalize_search_location_slots(self, query: StructuredQuery, user_text: str) -> None:
        district = query.hard.district
        if not isinstance(district, str) or not district.strip():
            return
        if query.hard.area is not None:
            return

        cleaned = district.strip()
        if self._looks_like_admin_division(cleaned):
            query.hard.district = cleaned[:-1] if cleaned.endswith("区") else cleaned
            return

        known_districts = await self._get_known_district_aliases()
        normalized = cleaned[:-1] if cleaned.endswith("区") else cleaned
        if normalized in known_districts or cleaned in known_districts:
            query.hard.district = normalized
            return

        if cleaned in user_text and not self._contains_explicit_admin_suffix_for(cleaned, user_text):
            query.hard.area = cleaned
            query.hard.district = None

    async def _get_known_district_aliases(self) -> set[str]:
        if self._known_district_aliases is not None:
            return self._known_district_aliases

        aliases: set[str] = set()
        client = self.landmarks_client
        list_fn = getattr(client, "list_landmarks", None) if client is not None else None
        if callable(list_fn):
            try:
                landmarks = await list_fn()
            except DataSourceError:
                landmarks = []
            for item in landmarks:
                district = getattr(item, "district", None)
                if not isinstance(district, str):
                    continue
                cleaned = district.strip()
                if not cleaned:
                    continue
                aliases.add(cleaned)
                if cleaned.endswith("区"):
                    aliases.add(cleaned[:-1])

        self._known_district_aliases = aliases
        return aliases

    @staticmethod
    def _contains_explicit_admin_suffix_for(token: str, text: str) -> bool:
        escaped = re.escape(token.strip())
        if not escaped:
            return False
        return bool(re.search(rf"{escaped}(新区|开发区|自治县|自治州|区|县|旗|市|州|盟)", text))

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

            if key in {"elevator", "value_for_money", "prioritize_subway_distance", "prefer_spacious", "prioritize_commute"}:
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
        if state.confirmed_constraints.area_min is not None:
            constraints.append(f"最小面积={int(state.confirmed_constraints.area_min)}")
        if state.confirmed_constraints.max_subway_dist is not None:
            constraints.append(f"地铁距离<={state.confirmed_constraints.max_subway_dist}")
        if state.confirmed_constraints.max_commute_min is not None:
            constraints.append(f"通勤<={state.confirmed_constraints.max_commute_min}")
        if state.soft_preferences.orientation:
            constraints.append(f"朝向偏好={state.soft_preferences.orientation}")
        if state.soft_preferences.prefer_spacious:
            constraints.append("偏好更大面积")
        if state.soft_preferences.prioritize_subway_distance:
            constraints.append("优先近地铁")
        if state.soft_preferences.prioritize_commute:
            constraints.append("优先短通勤")

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
