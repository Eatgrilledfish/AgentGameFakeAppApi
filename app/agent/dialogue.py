from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
import re
import time
from difflib import SequenceMatcher
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
    SessionHouseMemory,
    SoftPreferences,
    StructuredQuery,
    TagNeed,
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
    "VR",
    "AR",
    "线上看房",
    "线下看房",
    "养狗",
    "宠物",
    "遛狗",
    "金毛",
)
_STICKY_SOFT_BOOL_FIELDS = {"prefer_spacious", "prioritize_subway_distance", "prioritize_commute"}
_NEGATION_TOKENS = ("不", "不要", "不想", "不用", "别", "避免", "拒绝", "不希望", "不能")
_MUST_TOKENS = ("必须", "一定", "务必", "刚需", "只能", "只要", "得", "得要", "必要", "必须要", "一定要")
_TAG_TEXT_STOPWORDS = ("仅", "可", "需", "看房", "房", "租", "费用", "费", "押", "付", "支持")
_POSITIVE_TOKENS = ("要", "想", "希望", "能", "可以", "必须", "优先", "倾向")
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
    "subway_distance": "max_subway_dist",
    "max_distance": "max_distance",
}
_HOUSE_INTENTS = {
    IntentType.search,
    IntentType.compare,
    IntentType.house_detail,
    IntentType.amenities,
    IntentType.listings,
    IntentType.rent_check,
    IntentType.rent,
}
_TAG_CANCEL_TOKENS = ("无所谓", "不介意", "取消", "不用", "别管", "算了")
_TAG_CANCEL_TOPICS: dict[str, tuple[str, ...]] = {
    "物业": ("物业",),
    "宽带": ("宽带", "网费", "网络"),
    "暖气": ("暖气", "取暖"),
    "宠物": ("宠物", "养狗", "养猫"),
    "中介": ("中介",),
    "VR": ("vr", "线上看房", "在线看房"),
    "月付": ("月付",),
    "电梯": ("电梯",),
}
_REJECT_SCORE_PENALTY = 10.0
_RELEVANT_LEXICON_LIMIT = 60
_TAG_MATCH_SCORE_THRESHOLD = 0.46
_TAG_TOPIC_KEYWORDS = (
    "宽带",
    "物业",
    "暖气",
    "宠物",
    "养狗",
    "养猫",
    "中介",
    "月付",
    "押一",
    "押二",
    "押三",
    "电梯",
    "地铁",
    "看房",
    "vr",
    "ar",
    "合同",
    "停车",
    "采光",
    "朝南",
    "通勤",
    "民水民电",
    "商水商电",
)


@lru_cache(maxsize=1)
def _load_global_tag_catalog() -> tuple[str, ...]:
    candidate_paths: list[Path] = []
    env_path = os.getenv("AGENT_TAGS_PATH")
    if isinstance(env_path, str) and env_path.strip():
        configured = Path(env_path.strip())
        if not configured.is_absolute():
            configured = Path.cwd() / configured
        candidate_paths.append(configured)

    candidate_paths.extend(
        [
            Path.cwd() / "tags.txt",
            Path(__file__).resolve().parents[2] / "tags.txt",
        ]
    )

    text = ""
    for path in candidate_paths:
        try:
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    break
        except OSError:
            continue
    if not text.strip():
        return ()

    tags: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[,\n，、]+", text):
        token = " ".join(raw.strip().split())
        if not token or token in seen:
            continue
        seen.add(token)
        tags.append(token)
    return tuple(tags)


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
        self._known_landmark_aliases: set[str] | None = None
        self._global_tag_catalog: list[str] = list(_load_global_tag_catalog())

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

        llm_tag_need_locked = self._llm_parse_has_nonempty_tag_need(request.meta.get("llm_parse"))
        query = self._build_query(request, state)
        merged = self._merge_query_with_state(query, state, request.message)
        self._augment_tag_preferences_from_context(
            merged,
            state,
            request.message,
            preserve_existing_tag_need=llm_tag_need_locked,
        )
        if merged.intent == IntentType.chat:
            fallback_query = self.nlu.parse(request.message, state, request.case_type)
            self._augment_tag_preferences_from_context(
                fallback_query,
                state,
                request.message,
                preserve_existing_tag_need=llm_tag_need_locked,
            )
            if self._should_promote_chat_to_search_refinement(request.message, merged, fallback_query, state):
                log_event(
                    LOGGER,
                    "dialogue.intent.adjusted",
                    original_intent=IntentType.chat.value,
                    adjusted_intent=IntentType.search.value,
                    reason="preference_refinement_with_active_search_context",
                )
                merged = self._merge_query_with_state(fallback_query, state, request.message)
                merged.intent = IntentType.search
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
        if self._should_promote_detail_intent_to_search_refinement(request.message, merged, state):
            merged.intent = IntentType.search
            log_event(
                LOGGER,
                "dialogue.intent.adjusted",
                original_intent=IntentType.house_detail.value,
                adjusted_intent=IntentType.search.value,
                reason="preference_refinement_without_explicit_house_reference",
            )
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
                adjusted_intent=IntentType.rent_check.value,
                reason="rent_status_question_without_commitment",
            )
            merged.intent = IntentType.rent_check
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
            elif merged.intent == IntentType.rent_check:
                resp = await self._handle_rent_check(merged, state)
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
                    resp = InvokeResponse(
                        text=self._build_follow_up_text(follow_up),
                        clarify_questions=follow_up,
                        debug={"response_kind": "clarify"},
                    )
                else:
                    resp = await self._handle_search(merged, request, state)
            elif merged.intent == IntentType.chat:
                log_event(LOGGER, "dialogue.chat")
                self._remember_chat_preferences(state, merged, request.message)
                state.phase = SessionPhase.chatting
                resp = InvokeResponse(
                    text=self._fallback_chat_reply(request.message, merged),
                    debug={"response_kind": "chat"},
                )
            else:
                resp = await self._handle_search(merged, request, state)
        except DataSourceError as exc:
            log_event(LOGGER, "dialogue.error", error=str(exc))
            LOGGER.exception("Data source error")
            resp = InvokeResponse(text=f"接口调用失败：{exc}，请稍后重试。", candidates=[], debug={"response_kind": "error"})

        resp.debug.setdefault("intent", merged.intent.value)
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
        for key in ("intent", "params", "tag_need"):
            if key in llm_parse:
                return True
        return False

    @staticmethod
    def _llm_parse_has_nonempty_tag_need(llm_parse: Any) -> bool:
        if not isinstance(llm_parse, dict):
            return False
        tag_need = llm_parse.get("tag_need")
        if not isinstance(tag_need, dict):
            return False
        for key in ("must", "avoid", "prefer"):
            values = tag_need.get(key)
            if not isinstance(values, list):
                continue
            if any(isinstance(item, str) and item.strip() for item in values):
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

    def _should_promote_chat_to_search_refinement(
        self,
        text: str,
        chat_query: StructuredQuery,
        fallback_query: StructuredQuery,
        state: SessionState,
    ) -> bool:
        if chat_query.intent != IntentType.chat:
            return False

        has_active_search_context = bool(state.last_top5) or bool(state.last_candidates) or state.phase in {
            SessionPhase.searching,
            SessionPhase.presenting,
            SessionPhase.refining,
        }
        if not has_active_search_context:
            return False

        if self._has_explicit_search_request(text):
            return True
        if self._text_requests_context_continuation(text):
            return True
        if self._text_has_direct_house_requirements(text):
            return True
        if self._text_has_explicit_search_constraints(text):
            return True

        if self._has_actionable_preferences(chat_query):
            return True
        if fallback_query.intent == IntentType.search and self._has_actionable_preferences(fallback_query):
            return True
        return False

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
    def _has_explicit_house_reference(text: str) -> bool:
        if _EXPLICIT_HOUSE_ID_PATTERN.search(text):
            return True
        if DialogueManager._extract_rank_index(text) is not None:
            return True
        return any(token in text for token in _HOUSE_REF_WORDS)

    def _should_promote_detail_intent_to_search_refinement(
        self,
        text: str,
        query: StructuredQuery,
        state: SessionState,
    ) -> bool:
        if query.intent not in {IntentType.house_detail, IntentType.listings}:
            return False

        has_active_search_context = bool(state.last_top5) or bool(state.last_candidates) or state.phase in {
            SessionPhase.searching,
            SessionPhase.presenting,
            SessionPhase.refining,
        }
        if not has_active_search_context:
            return False
        if self._has_explicit_house_reference(text):
            return False

        # LLM-first strategy: if parse carries actionable constraints/preferences
        # (excluding explicit house refs), treat as search refinement.
        hard = HardConstraints.model_validate(query.hard.model_dump())
        hard.house_id = None
        hard.listing_platform = None
        inferred = StructuredQuery(
            intent=IntentType.search,
            hard=hard,
            soft=SoftPreferences.model_validate(query.soft.model_dump()),
            confidence=query.confidence,
        )
        if not self._has_actionable_preferences(inferred):
            return False
        return True

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
        state.candidate_state.focus_house_id = recommend.house_id
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
        self._normalize_query(query, request.message)
        self._refresh_session_req(state, query, request.message)

        plan = await self.planner.build_plan(query)
        log_event(LOGGER, "dialogue.plan.built", plan_type=plan.plan_type, landmark_id=plan.landmark_id)
        candidates = await self.planner.execute_plan(plan, query, request.case_type)
        log_event(LOGGER, "dialogue.plan.executed", candidate_count=len(candidates))

        filtered = self._filter_candidates_with_hard_constraints(candidates, query)
        if len(filtered) < 5:
            expanded = self._expand_budget_and_refilter(candidates, query, filtered)
            if expanded:
                filtered = expanded

        topk = self._pick_topk_candidates(filtered, query, limit=15)
        self._update_tag_lexicon_memory(state, topk)
        allowlist = {house.house_id for house in topk}
        should_run_tag_filter = self._should_run_tag_semantic_filter(query, state, topk)
        semantic = self._run_tag_semantic_filter(query, state, topk, allowlist=allowlist, enabled=should_run_tag_filter)

        rank_query = query.model_copy(deep=True)
        if rank_query.hard.district and "," in rank_query.hard.district:
            # Multi-district has been applied in pre-filtering.
            rank_query.hard.district = None
        deterministic_ranked = await self.ranker.rank_two_stage(topk, rank_query, max_output=max(1, len(topk)))
        fused_ranked, semantic_fusion = self._apply_semantic_decisions(
            ranked_views=deterministic_ranked,
            candidates=topk,
            query=rank_query,
            semantic=semantic,
        )
        top = fused_ranked[: self.max_output_candidates]
        if not top:
            top = deterministic_ranked[: self.max_output_candidates]
            semantic_fusion["fallback"] = "deterministic_topk"
        log_event(LOGGER, "dialogue.rank.done", top_count=len(top))

        state.phase = SessionPhase.presenting
        state.confirmed_constraints = query.hard
        state.soft_preferences = query.soft
        context_candidates = topk
        merged_need_payload = semantic.get("merged_tag_need")
        has_tag_need = False
        if isinstance(merged_need_payload, dict):
            has_tag_need = any(
                isinstance(merged_need_payload.get(key), list) and bool(merged_need_payload.get(key))
                for key in ("must", "avoid", "prefer")
            )
        if has_tag_need:
            selected_set = {
                house_id
                for house_id in semantic.get("selected", [])
                if isinstance(house_id, str) and house_id
            }
            context_candidates = [house for house in topk if house.house_id in selected_set]

        compact_candidates = self._compress_candidates(context_candidates)
        state.last_candidates = compact_candidates
        state.house_context_top10 = compact_candidates[:10]
        state.last_top5 = top
        self._remember_search_snapshot(state, request.message, query)

        return self.formatter.render(
            case_type=request.case_type,
            query=query,
            top_houses=top,
            debug={
                "response_kind": "search",
                "plan": plan.plan_type,
                "candidate_count": len(candidates),
                "filtered_count": len(filtered),
                "topk_count": len(topk),
                "context_candidate_count": len(context_candidates),
                "semantic_filter": semantic,
                "semantic_fusion": semantic_fusion,
            },
        )

    def _normalize_query(self, query: StructuredQuery, user_text: str) -> None:
        hard = query.hard
        if hard.district:
            hard.district = self._normalize_district_value(hard.district)
        if hard.layout:
            hard.layout = self._normalize_layout_value(hard.layout)
        if hard.budget_min is not None and hard.budget_max is not None and hard.budget_min > hard.budget_max:
            hard.budget_min, hard.budget_max = hard.budget_max, hard.budget_min
        if hard.budget_max is not None and hard.budget_min is None and self._text_implies_budget_range(user_text):
            hard.budget_min = int(max(0, round(hard.budget_max * 0.9)))
            hard.budget_max = int(round(hard.budget_max * 1.1))
        if hard.max_subway_dist is None and any(token in user_text for token in ("近地铁", "离地铁近", "靠近地铁")):
            hard.max_subway_dist = 800
        if query.soft.decoration == "精装修":
            query.soft.decoration = "精装"
        if query.soft.decoration == "简装修":
            query.soft.decoration = "简装"
        query.tag_need = self._normalize_tag_need(query.tag_need, query.soft)

    @staticmethod
    def _normalize_district_value(value: str) -> str | None:
        tokens = [item.strip() for item in re.split(r"[、,，/和\s]+", value) if item.strip()]
        if not tokens:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            cleaned = token
            for suffix in ("区", "县", "市"):
                if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                    cleaned = cleaned[: -len(suffix)]
                    break
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        if not normalized:
            return None
        return ",".join(normalized)

    @staticmethod
    def _text_implies_budget_range(text: str) -> bool:
        if any(token in text for token in ("左右", "上下", "约", "大概")):
            return True
        if "附近" not in text:
            return False
        if re.search(r"(预算|价格|租金|房租).{0,6}附近", text):
            return True
        if re.search(r"\d+\s*(?:k|K|千|元|块).{0,3}附近", text):
            return True
        return False

    @staticmethod
    def _normalize_layout_value(value: str) -> str:
        text = value.strip()
        if not text:
            return value
        trans = str.maketrans({"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5"})
        normalized = text.translate(trans)
        range_match = re.search(r"([1-5])\s*[-~到至]\s*([1-5])\s*(?:居|室)", normalized)
        if range_match:
            a = int(range_match.group(1))
            b = int(range_match.group(2))
            low, high = sorted((a, b))
            return f"{low},{high}居"
        single_match = re.search(r"([1-9])\s*(?:居|室)", normalized)
        if single_match:
            return f"{single_match.group(1)}居"
        return normalized

    @staticmethod
    def _normalize_tag_need(tag_need: TagNeed, soft: SoftPreferences) -> TagNeed:
        must = [item.strip() for item in tag_need.must if isinstance(item, str) and item.strip()]
        avoid = [item.strip() for item in tag_need.avoid if isinstance(item, str) and item.strip()]
        prefer = [item.strip() for item in tag_need.prefer if isinstance(item, str) and item.strip()]
        for item in soft.avoid_tags:
            if item and item not in avoid:
                avoid.append(item)
        for item in soft.preferred_tags:
            if item and item not in prefer:
                prefer.append(item)
        return TagNeed(must=must[:8], avoid=avoid[:12], prefer=prefer[:12])

    def _refresh_session_req(self, state: SessionState, query: StructuredQuery, user_text: str) -> None:
        merged_hard = HardConstraints.model_validate(state.req.hard.model_dump())
        for key, value in query.hard.model_dump(exclude_none=True).items():
            setattr(merged_hard, key, value)
        state.req.hard = merged_hard

        merged_tag_need = self._merge_tag_need_accumulated(
            current=self._normalize_tag_need(query.tag_need, query.soft),
            accumulated=state.req.soft.tag_need_accumulated,
            user_text=user_text,
        )
        state.req.soft.tag_need_accumulated = merged_tag_need
        notes = list(state.req.soft.notes)
        for bucket in (merged_tag_need.must, merged_tag_need.avoid, merged_tag_need.prefer):
            for item in bucket:
                note = item.strip()
                if note and note not in notes:
                    notes.append(note)
        state.req.soft.notes = notes[-30:]

    def _merge_tag_need_accumulated(self, *, current: TagNeed, accumulated: TagNeed, user_text: str) -> TagNeed:
        merged_must = list(dict.fromkeys([*accumulated.must, *current.must]))
        merged_avoid = list(dict.fromkeys([*accumulated.avoid, *current.avoid]))
        merged_prefer = list(dict.fromkeys([*accumulated.prefer, *current.prefer]))
        cleaned = TagNeed(must=merged_must[:20], avoid=merged_avoid[:20], prefer=merged_prefer[:20])
        return self._apply_tag_need_revocations(cleaned, user_text)

    @staticmethod
    def _apply_tag_need_revocations(tag_need: TagNeed, user_text: str) -> TagNeed:
        text = (user_text or "").lower()
        if not text or not any(token in text for token in _TAG_CANCEL_TOKENS):
            return tag_need

        revoked_keywords: set[str] = set()
        for _, aliases in _TAG_CANCEL_TOPICS.items():
            for alias in aliases:
                if alias.lower() in text:
                    revoked_keywords.update(a.lower() for a in aliases)
                    break

        if not revoked_keywords:
            return tag_need

        def keep(item: str) -> bool:
            lowered = item.lower()
            return not any(keyword in lowered for keyword in revoked_keywords)

        return TagNeed(
            must=[item for item in tag_need.must if keep(item)],
            avoid=[item for item in tag_need.avoid if keep(item)],
            prefer=[item for item in tag_need.prefer if keep(item)],
        )

    def _filter_candidates_with_hard_constraints(self, candidates: list[HouseLite], query: StructuredQuery) -> list[HouseLite]:
        return [house for house in candidates if self._matches_hard_constraints(house, query)]

    def _expand_budget_and_refilter(
        self,
        candidates: list[HouseLite],
        query: StructuredQuery,
        base_filtered: list[HouseLite],
    ) -> list[HouseLite]:
        if query.hard.budget_max is None:
            return base_filtered
        if len(base_filtered) >= 5:
            return base_filtered

        relaxed = list(base_filtered)
        budget_max = query.hard.budget_max
        for _ in range(2):
            budget_max = max(int(round(budget_max * 1.1)), budget_max + 300)
            query_copy = query.model_copy(deep=True)
            query_copy.hard.budget_max = budget_max
            rows = [house for house in candidates if self._matches_hard_constraints(house, query_copy)]
            if len(rows) > len(relaxed):
                relaxed = rows
            if len(relaxed) >= 5:
                break
        return relaxed

    def _matches_hard_constraints(self, house: HouseLite, query: StructuredQuery) -> bool:
        hard = query.hard
        if hard.budget_max is not None and house.rent is not None and house.rent > hard.budget_max:
            return False
        if hard.budget_min is not None and house.rent is not None and house.rent < hard.budget_min:
            return False
        if hard.max_subway_dist is not None and house.subway_distance is not None and house.subway_distance > hard.max_subway_dist:
            return False
        if hard.area_min is not None and house.area is not None and house.area < hard.area_min:
            return False
        if hard.district and house.district:
            district_values = {item.strip() for item in hard.district.split(",") if item.strip()}
            if district_values and house.district not in district_values:
                return False
        if hard.rent_type and house.layout:
            if hard.rent_type == "整租" and any(token in house.layout for token in ("单间", "合租")):
                return False
            if hard.rent_type == "合租" and not any(token in house.layout for token in ("单间", "合租")):
                return False
        bedroom_tokens = self._extract_bedroom_tokens(hard.layout)
        if bedroom_tokens and house.layout:
            house_bedrooms = self._extract_bedroom_tokens(house.layout)
            if house_bedrooms and not (house_bedrooms & bedroom_tokens):
                return False
        if query.soft.decoration and house.decoration:
            if query.soft.decoration != house.decoration:
                return False
        if query.soft.elevator is not None and house.elevator is not None:
            if query.soft.elevator != house.elevator:
                return False
        return True

    @staticmethod
    def _extract_bedroom_tokens(value: str | None) -> set[str]:
        if not value:
            return set()
        normalized = value.translate(str.maketrans({"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5"}))
        values = set(re.findall(r"([1-9])\s*(?:居|室)?", normalized))
        return values

    def _pick_topk_candidates(self, candidates: list[HouseLite], query: StructuredQuery, *, limit: int) -> list[HouseLite]:
        ranked = sorted(candidates, key=lambda house: self._base_house_score(house, query), reverse=True)
        return ranked[:limit]

    def _base_house_score(self, house: HouseLite, query: StructuredQuery) -> float:
        score = 0.0
        if house.rent is not None:
            score -= house.rent / 1000.0
            if query.hard.budget_max is not None:
                score += max(0.0, (query.hard.budget_max - house.rent) / max(query.hard.budget_max, 1))
        if house.subway_distance is not None:
            score += max(0.0, 2.0 - house.subway_distance / 1000.0)
        if house.commute_to_xierqi_min is not None:
            score += max(0.0, 2.0 - house.commute_to_xierqi_min / 60.0)
        if house.area is not None:
            score += min(2.0, house.area / 80.0)
        if query.hard.rent_type == "整租" and house.layout and not any(token in house.layout for token in ("单间", "合租")):
            score += 0.5
        return score

    def _update_tag_lexicon_memory(self, state: SessionState, candidates: list[HouseLite]) -> None:
        if not state.reverse_lexicon and state.tag_lexicon:
            state.reverse_lexicon = {tag: tid for tid, tag in state.tag_lexicon.items()}

        for house in candidates:
            tag_ids: list[str] = []
            for raw_tag in house.tags:
                if not isinstance(raw_tag, str):
                    continue
                tag = raw_tag.strip()
                if not tag:
                    continue
                tid = state.reverse_lexicon.get(tag)
                if tid is None:
                    tid = f"t{len(state.tag_lexicon) + 1}"
                    state.tag_lexicon[tid] = tag
                    state.reverse_lexicon[tag] = tid
                if tid not in tag_ids:
                    tag_ids.append(tid)

            state.houses[house.house_id] = SessionHouseMemory(
                tag_ids=sorted(tag_ids),
                price=house.rent,
                subway_distance=house.subway_distance,
                rental_type=self._infer_rental_type(house),
                area_sqm=house.area,
                updated_ts=int(time.time()),
            )

        house_ids = [house.house_id for house in candidates]
        state.candidate_state.latest_house_ids = house_ids
        if house_ids:
            state.candidate_state.focus_house_id = house_ids[0]

    @staticmethod
    def _infer_rental_type(house: HouseLite) -> str | None:
        if not house.layout:
            return None
        if any(token in house.layout for token in ("单间", "合租")):
            return "合租"
        return "整租"

    def _should_run_tag_semantic_filter(self, query: StructuredQuery, state: SessionState, candidates: list[HouseLite]) -> bool:
        if not candidates:
            return False
        merged = self._build_merged_tag_need(query, state)
        return bool(merged.must or merged.avoid or merged.prefer)

    def _build_merged_tag_need(self, query: StructuredQuery, state: SessionState) -> TagNeed:
        current = query.tag_need
        accumulated = state.req.soft.tag_need_accumulated
        return TagNeed(
            must=list(dict.fromkeys([*accumulated.must, *current.must]))[:20],
            avoid=list(dict.fromkeys([*accumulated.avoid, *current.avoid]))[:20],
            prefer=list(dict.fromkeys([*accumulated.prefer, *current.prefer]))[:20],
        )

    def _run_tag_semantic_filter(
        self,
        query: StructuredQuery,
        state: SessionState,
        candidates: list[HouseLite],
        *,
        allowlist: set[str],
        enabled: bool,
    ) -> dict[str, Any]:
        merged_need = self._build_merged_tag_need(query, state)
        need_terms = [*merged_need.must, *merged_need.avoid, *merged_need.prefer]
        if not enabled or not need_terms:
            return {
                "enabled": enabled,
                "candidate_filter_ran": False,
                "merged_tag_need": merged_need.model_dump(),
                "relevant_tag_ids": [],
                "selected": [house.house_id for house in candidates if house.house_id in allowlist],
                "must_confirm": [],
                "rejected": [],
            }

        relevant_tag_ids = self._tag_focus(need_terms, state.tag_lexicon)
        relevant_lexicon = self._build_relevant_lexicon_for_candidate_filter(
            relevant_tag_ids=relevant_tag_ids,
            candidates=candidates,
            state=state,
            merged_need=merged_need,
            limit=_RELEVANT_LEXICON_LIMIT,
        )
        if not relevant_lexicon:
            fallback = self._filter_candidates_by_raw_tags(candidates, allowlist=allowlist, merged_need=merged_need)
            return {
                "enabled": True,
                "candidate_filter_ran": True,
                "merged_tag_need": merged_need.model_dump(),
                "relevant_tag_ids": [],
                "selected": fallback["selected"],
                "must_confirm": fallback["must_confirm"],
                "rejected": fallback["rejected"],
            }

        selected: list[str] = []
        must_confirm: list[dict[str, str]] = []
        rejected: list[dict[str, str]] = []
        relevant_tag_id_set = set(relevant_lexicon.keys())

        for house in candidates:
            if house.house_id not in allowlist:
                continue
            memory = state.houses.get(house.house_id)
            if memory is None:
                must_confirm.append({"house_id": house.house_id, "reason": "标签信息缺失，无法匹配"})
                continue
            house_tag_ids = [tid for tid in memory.tag_ids if tid in relevant_tag_id_set][:20]
            tag_texts = [relevant_lexicon.get(tid, "") for tid in house_tag_ids if relevant_lexicon.get(tid)]

            avoid_hit = self._match_any_need(merged_need.avoid, tag_texts)
            if avoid_hit is not None:
                rejected.append({"house_id": house.house_id, "reason": f"命中避坑标签：{avoid_hit}"})
                continue

            must_hit = self._match_any_need(merged_need.must, tag_texts)
            if merged_need.must and must_hit is None:
                must_confirm.append({"house_id": house.house_id, "reason": "刚需标签未明确标注"})
                continue

            prefer_hit = self._match_any_need(merged_need.prefer, tag_texts)
            if not merged_need.must and merged_need.prefer and prefer_hit is None:
                rejected.append({"house_id": house.house_id, "reason": "偏好标签未命中"})
                continue

            selected.append(house.house_id)

        sanitized = self._sanitize_candidate_filter_output(
            allowlist=allowlist,
            selected=selected,
            must_confirm=must_confirm,
            rejected=rejected,
        )
        return {
            "enabled": True,
            "candidate_filter_ran": True,
            "merged_tag_need": merged_need.model_dump(),
            "relevant_tag_ids": list(relevant_lexicon.keys()),
            "selected": sanitized["selected"],
            "must_confirm": sanitized["must_confirm"],
            "rejected": sanitized["rejected"],
        }

    def _filter_candidates_by_raw_tags(
        self,
        candidates: list[HouseLite],
        *,
        allowlist: set[str],
        merged_need: TagNeed,
    ) -> dict[str, Any]:
        selected: list[str] = []
        must_confirm: list[dict[str, str]] = []
        rejected: list[dict[str, str]] = []

        for house in candidates:
            if house.house_id not in allowlist:
                continue
            raw_tags = [tag.strip() for tag in house.tags if isinstance(tag, str) and tag.strip()]
            if not raw_tags:
                must_confirm.append({"house_id": house.house_id, "reason": "标签信息缺失，无法匹配"})
                continue

            avoid_hit = self._match_any_need(merged_need.avoid, raw_tags)
            if avoid_hit is not None:
                rejected.append({"house_id": house.house_id, "reason": f"命中避坑标签：{avoid_hit}"})
                continue

            must_hit = self._match_any_need(merged_need.must, raw_tags)
            if merged_need.must and must_hit is None:
                must_confirm.append({"house_id": house.house_id, "reason": "刚需标签未明确标注"})
                continue

            prefer_hit = self._match_any_need(merged_need.prefer, raw_tags)
            if not merged_need.must and merged_need.prefer and prefer_hit is None:
                rejected.append({"house_id": house.house_id, "reason": "偏好标签未命中"})
                continue

            selected.append(house.house_id)

        return self._sanitize_candidate_filter_output(
            allowlist=allowlist,
            selected=selected,
            must_confirm=must_confirm,
            rejected=rejected,
        )

    def _build_relevant_lexicon_for_candidate_filter(
        self,
        *,
        relevant_tag_ids: list[str],
        candidates: list[HouseLite],
        state: SessionState,
        merged_need: TagNeed,
        limit: int,
    ) -> dict[str, str]:
        if not relevant_tag_ids:
            return {}
        candidate_coverage: dict[str, int] = {tid: 0 for tid in relevant_tag_ids}
        for house in candidates:
            memory = state.houses.get(house.house_id)
            if not memory:
                continue
            for tid in memory.tag_ids:
                if tid in candidate_coverage:
                    candidate_coverage[tid] += 1

        need_terms = [*merged_need.must, *merged_need.avoid, *merged_need.prefer]
        ranked = sorted(
            [tid for tid in relevant_tag_ids if tid in state.tag_lexicon],
            key=lambda tid: (
                candidate_coverage.get(tid, 0),
                max((SequenceMatcher(None, need, state.tag_lexicon.get(tid, "")).ratio() for need in need_terms), default=0.0),
            ),
            reverse=True,
        )
        clipped = ranked[:limit]
        return {tid: state.tag_lexicon[tid] for tid in clipped}

    @staticmethod
    def _sanitize_candidate_filter_output(
        *,
        allowlist: set[str],
        selected: list[str],
        must_confirm: list[dict[str, str]],
        rejected: list[dict[str, str]],
    ) -> dict[str, Any]:
        selected_clean = [house_id for house_id in selected if house_id in allowlist]
        must_confirm_clean = [
            item
            for item in must_confirm
            if isinstance(item, dict) and isinstance(item.get("house_id"), str) and item["house_id"] in allowlist
        ]
        rejected_clean = [
            item
            for item in rejected
            if isinstance(item, dict) and isinstance(item.get("house_id"), str) and item["house_id"] in allowlist
        ]
        return {"selected": selected_clean, "must_confirm": must_confirm_clean, "rejected": rejected_clean}

    def _apply_semantic_decisions(
        self,
        *,
        ranked_views: list[Any],
        candidates: list[HouseLite],
        query: StructuredQuery,
        semantic: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        if not ranked_views:
            return ranked_views, {"applied": False}

        merged_need = TagNeed.model_validate(semantic.get("merged_tag_need") or {"must": [], "avoid": [], "prefer": []})
        selected_ids = set(str(item) for item in semantic.get("selected", []))
        must_confirm_map: dict[str, str] = {}
        for row in semantic.get("must_confirm", []):
            if isinstance(row, dict):
                house_id = row.get("house_id")
                reason = row.get("reason")
                if isinstance(house_id, str) and isinstance(reason, str):
                    must_confirm_map[house_id] = reason
        rejected_map: dict[str, str] = {}
        for row in semantic.get("rejected", []):
            if isinstance(row, dict):
                house_id = row.get("house_id")
                reason = row.get("reason")
                if isinstance(house_id, str) and isinstance(reason, str):
                    rejected_map[house_id] = reason

        top2_ids = {view.house_id for view in ranked_views[:2]}
        candidate_map = {house.house_id: house for house in candidates}
        decisions: dict[str, dict[str, Any]] = {}

        for view in ranked_views:
            base_score = float(view.score or 0.0)
            house_id = view.house_id
            adjusted = base_score
            action = "neutral"
            reason = ""
            hard_drop = False

            if house_id in selected_ids:
                adjusted += 15.0
                action = "selected_boost"
            if house_id in must_confirm_map:
                adjusted += 3.0
                action = "must_confirm_boost"
                reason = must_confirm_map[house_id]

            if house_id in rejected_map:
                reject_reason = rejected_map[house_id]
                is_hard = self._is_hard_conflict(reject_reason, merged_need)
                protected_top2 = (
                    house_id in top2_ids
                    and house_id in candidate_map
                    and self._matches_hard_constraints(candidate_map[house_id], query)
                )
                if is_hard and not protected_top2:
                    hard_drop = True
                    action = "rejected_drop"
                    reason = reject_reason
                else:
                    adjusted -= _REJECT_SCORE_PENALTY
                    action = "rejected_penalty"
                    reason = reject_reason

            decisions[house_id] = {
                "base_score": round(base_score, 4),
                "adjusted_score": round(adjusted, 4),
                "action": action,
                "reason": reason,
                "hard_drop": hard_drop,
                "needs_confirm": house_id in must_confirm_map,
            }

        kept = [view for view in ranked_views if not decisions.get(view.house_id, {}).get("hard_drop", False)]
        kept = sorted(kept, key=lambda view: decisions.get(view.house_id, {}).get("adjusted_score", float(view.score or 0.0)), reverse=True)
        return kept, {"applied": True, "decisions": decisions, "must_confirm": must_confirm_map}

    def _is_hard_conflict(self, rejected_reason: str, tag_need: TagNeed) -> bool:
        reason = re.sub(r"[^\w\u4e00-\u9fa5]+", "", rejected_reason.lower())
        if not reason:
            return False
        for item in tag_need.avoid:
            normalized = re.sub(r"[^\w\u4e00-\u9fa5]+", "", item.lower())
            if normalized and (normalized in reason or reason in normalized):
                return True
        for item in tag_need.must:
            normalized = re.sub(r"[^\w\u4e00-\u9fa5]+", "", item.lower())
            if normalized and (normalized in reason or reason in normalized):
                return True
        return False

    def _tag_focus(self, needs: list[str], lexicon: dict[str, str]) -> list[str]:
        picked: list[str] = []
        for tid, tag in lexicon.items():
            if any(self._semantic_match(need, tag) for need in needs):
                picked.append(tid)
        return picked

    def _match_any_need(self, needs: list[str], tags: list[str]) -> str | None:
        for need in needs:
            for tag in tags:
                if self._semantic_match(need, tag):
                    return tag
        return None

    @staticmethod
    def _semantic_match(need: str, tag: str) -> bool:
        left = re.sub(r"[^\w\u4e00-\u9fa5]+", "", need.lower())
        right = re.sub(r"[^\w\u4e00-\u9fa5]+", "", tag.lower())
        if not left or not right:
            return False
        if DialogueManager._has_tag_polarity_conflict(left, right):
            return False
        if left in right or right in left:
            return True
        # Prefer explicit topic overlap to avoid fuzzy-ratio misses such as "宽带包含" vs "宽带标签X".
        for keyword in _TAG_TOPIC_KEYWORDS:
            if keyword in left and keyword in right:
                return True
        return SequenceMatcher(None, left, right).ratio() >= 0.54

    @staticmethod
    def _has_tag_polarity_conflict(left: str, right: str) -> bool:
        dog_tokens = ("养狗", "小型犬")
        cat_tokens = ("养猫",)
        if any(token in left for token in dog_tokens) and any(token in right for token in cat_tokens):
            return True
        if any(token in left for token in cat_tokens) and any(token in right for token in dog_tokens):
            return True

        allow_pet_tokens = ("可养宠", "可养狗", "可养猫", "宠物友好", "仅限小型犬")
        deny_pet_tokens = ("不可养宠", "禁止养宠", "不允许养宠", "不可养宠物")
        if any(token in left for token in allow_pet_tokens) and any(token in right for token in deny_pet_tokens):
            return True
        if any(token in left for token in deny_pet_tokens) and any(token in right for token in allow_pet_tokens):
            return True
        return False

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
        state.candidate_state.focus_house_id = house_id
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
            state.candidate_state.focus_house_id = referenced[0]

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
            state.candidate_state.focus_house_id = house_id
            return InvokeResponse(
                text=f"{house_id} 暂未查到平台挂牌记录。",
                debug={"response_kind": "detail", "referenced_house_ids": [house_id]},
            )

        state.focus_house_id = house_id
        state.candidate_state.focus_house_id = house_id
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

    async def _handle_rent_check(self, query: StructuredQuery, state: SessionState) -> InvokeResponse:
        house_id = query.hard.house_id or state.focus_house_id or state.candidate_state.focus_house_id
        if not house_id:
            return InvokeResponse(
                text="请告诉我要确认的房源（例如 HF_2001，或说第一套/这套）。",
                debug={"response_kind": "detail"},
            )
        allowlist = self._build_action_house_allowlist(state)
        if allowlist and house_id not in allowlist:
            fallback = [hid for hid in state.candidate_state.latest_house_ids if hid in allowlist][:5]
            return InvokeResponse(
                text=f"未在当前候选中找到 {house_id}，请从最近推荐房源中选择。",
                debug={"response_kind": "detail", "referenced_house_ids": fallback},
            )

        detail = await self._load_house_detail(house_id)
        if detail is None:
            return InvokeResponse(
                text=f"未找到房源 {house_id} 的状态，请确认 house_id 是否正确。",
                debug={"response_kind": "detail", "referenced_house_ids": [house_id]},
            )

        listings = await self._load_listings(house_id)
        listing_status = next((row.status for row in listings if row.status), None)
        status = detail.status or listing_status or "状态未知"
        state.focus_house_id = house_id
        state.candidate_state.focus_house_id = house_id
        return InvokeResponse(
            text=f"{house_id} 当前状态：{status}。",
            debug={"response_kind": "detail", "referenced_house_ids": [house_id]},
        )

    async def _handle_action(self, query: StructuredQuery, state: SessionState) -> InvokeResponse:
        house_id = query.hard.house_id or state.focus_house_id or state.candidate_state.focus_house_id
        if not house_id:
            return InvokeResponse(
                text="请提供要操作的房源 house_id（例如 HF_2001），或先说“租第一套”。",
                debug={"response_kind": "action"},
            )
        allowlist = self._build_action_house_allowlist(state)
        if allowlist and house_id not in allowlist:
            fallback = [hid for hid in state.candidate_state.latest_house_ids if hid in allowlist][:5]
            return InvokeResponse(
                text=f"未在当前候选中找到 {house_id}，请先指定最近推荐的房源再操作。",
                debug={"response_kind": "action", "referenced_house_ids": fallback},
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
        state.candidate_state.focus_house_id = house_id
        state.focus_listing_platform = platform
        log_event(LOGGER, "dialogue.action.done", action=action, result=result)
        return InvokeResponse(
            text=f"{message}：{house_id}（{platform.value}）。",
            debug={"response_kind": "action", "action_result": result, "referenced_house_ids": [house_id]},
        )

    @staticmethod
    def _build_action_house_allowlist(state: SessionState) -> set[str]:
        allowlist = {
            house_id
            for house_id in state.candidate_state.latest_house_ids
            if isinstance(house_id, str) and house_id
        }
        # Keep current focus actionable even when user pivots back to an earlier recommended house.
        if isinstance(state.candidate_state.focus_house_id, str) and state.candidate_state.focus_house_id:
            allowlist.add(state.candidate_state.focus_house_id)
        if isinstance(state.focus_house_id, str) and state.focus_house_id:
            allowlist.add(state.focus_house_id)
        if not allowlist and state.last_top5:
            allowlist = {item.house_id for item in state.last_top5 if isinstance(item.house_id, str) and item.house_id}
        return allowlist

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
        tag_need = query.tag_need
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
                bool(soft.preferred_tags),
                bool(soft.avoid_tags),
                soft.prefer_spacious,
                soft.prioritize_subway_distance,
                soft.prioritize_commute,
                bool(tag_need.must),
                bool(tag_need.avoid),
                bool(tag_need.prefer),
            ]
        )

    def _augment_tag_preferences_from_context(
        self,
        query: StructuredQuery,
        state: SessionState,
        user_text: str,
        *,
        preserve_existing_tag_need: bool = False,
    ) -> None:
        if preserve_existing_tag_need:
            return
        text = user_text.strip()
        if not text:
            return

        tag_pool = self._collect_tag_pool_for_inference(state)
        if not tag_pool:
            return

        must_tags, preferred_tags, avoid_tags = self._infer_tag_need_from_text(text, tag_pool)
        if not must_tags and not preferred_tags and not avoid_tags:
            return

        def merge_unique(base: list[str], extra: list[str], *, limit: int) -> list[str]:
            merged: list[str] = []
            seen: set[str] = set()
            for raw in [*base, *extra]:
                if not isinstance(raw, str):
                    continue
                item = raw.strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                merged.append(item)
                if len(merged) >= limit:
                    break
            return merged

        merged_must = merge_unique(query.tag_need.must, must_tags, limit=20)
        merged_avoid = merge_unique(query.tag_need.avoid, avoid_tags, limit=20)
        merged_prefer = merge_unique(query.tag_need.prefer, preferred_tags, limit=20)
        merged_prefer = [item for item in merged_prefer if item not in merged_must and item not in merged_avoid]
        query.tag_need = TagNeed(must=merged_must, avoid=merged_avoid, prefer=merged_prefer[:20])

        merged_soft_avoid = merge_unique(query.soft.avoid_tags, merged_avoid, limit=20)
        merged_soft_preferred = merge_unique(query.soft.preferred_tags, [*merged_must, *merged_prefer], limit=20)
        merged_soft_preferred = [item for item in merged_soft_preferred if item not in merged_soft_avoid]
        query.soft.preferred_tags = merged_soft_preferred[:20]
        query.soft.avoid_tags = merged_soft_avoid[:20]
        log_event(
            LOGGER,
            "dialogue.tag_preferences.augmented",
            must_tags=query.tag_need.must[:8],
            preferred_tags=query.tag_need.prefer[:8],
            avoid_tags=query.tag_need.avoid[:8],
        )

    def _collect_tag_pool_for_inference(self, state: SessionState, *, max_tags: int = 400) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for raw in self._global_tag_catalog:
            if not isinstance(raw, str):
                continue
            item = raw.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            tags.append(item)
            if len(tags) >= max_tags:
                return tags

        for raw in self._collect_context_tags(state, max_tags=max_tags):
            if not isinstance(raw, str):
                continue
            item = raw.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            tags.append(item)
            if len(tags) >= max_tags:
                break
        return tags

    @staticmethod
    def _collect_context_tags(state: SessionState, *, max_tags: int = 200) -> list[str]:
        seen: set[str] = set()
        tags: list[str] = []

        def collect(values: list[Any] | None) -> None:
            if not isinstance(values, list):
                return
            for raw in values:
                if not isinstance(raw, str):
                    continue
                tag = raw.strip()
                if not tag or tag in seen:
                    continue
                seen.add(tag)
                tags.append(tag)
                if len(tags) >= max_tags:
                    return

        for house in state.house_context_top10[:40]:
            collect(getattr(house, "tags", []))
            if len(tags) >= max_tags:
                return tags
        for house in state.last_candidates[:40]:
            collect(getattr(house, "tags", []))
            if len(tags) >= max_tags:
                return tags
        for house in state.last_top5[:5]:
            collect(getattr(house, "tags", []))
            if len(tags) >= max_tags:
                return tags
        return tags

    @staticmethod
    def _infer_tag_need_from_text(text: str, tag_pool: list[str]) -> tuple[list[str], list[str], list[str]]:
        normalized_text = DialogueManager._normalize_tag_text(text)
        if not normalized_text:
            return [], [], []

        must: list[str] = []
        preferred: list[str] = []
        avoid: list[str] = []

        for tag in tag_pool:
            score, matched_signals = DialogueManager._match_tag_signal_score(normalized_text, tag)
            if score < _TAG_MATCH_SCORE_THRESHOLD:
                continue

            if DialogueManager._is_negative_intent_for_tag(normalized_text, tag, matched_signals):
                if tag not in avoid:
                    avoid.append(tag)
            elif DialogueManager._is_must_intent_for_tag(normalized_text, tag, matched_signals):
                if tag not in must:
                    must.append(tag)
            else:
                if tag not in preferred:
                    preferred.append(tag)

        # If both sides include same tag, keep explicit avoid only.
        must = [tag for tag in must if tag not in avoid]
        preferred = [tag for tag in preferred if tag not in avoid and tag not in must]
        return must[:10], preferred[:12], avoid[:12]

    @staticmethod
    def _match_tag_signal_score(text: str, tag: str) -> tuple[float, list[str]]:
        normalized_tag = DialogueManager._normalize_tag_text(tag)
        if not normalized_tag:
            return 0.0, []

        signals = DialogueManager._extract_tag_signals(tag)
        matched = [sig for sig in signals if len(sig) >= 2 and sig in text]

        ratio = SequenceMatcher(None, normalized_tag, text).ratio()
        if normalized_tag in text:
            ratio = max(ratio, 0.78)
        if matched:
            ratio = max(ratio, min(0.95, 0.56 + 0.06 * len(matched) + 0.05 * max(len(sig) for sig in matched)))

        return ratio, matched

    @staticmethod
    def _is_negative_intent_for_tag(text: str, tag: str, matched_signals: list[str]) -> bool:
        tag_norm = DialogueManager._normalize_tag_text(tag)
        if not tag_norm:
            return False

        signals = matched_signals or [sig for sig in DialogueManager._extract_tag_signals(tag) if len(sig) >= 2]
        has_positive = False
        question_markers = ("能不能", "可不可以", "能否", "是否可以", "行不行")
        for signal in signals:
            start = text.find(signal)
            if start == -1:
                continue
            left = text[max(0, start - 4) : start]
            right = text[start + len(signal) : start + len(signal) + 4]
            if any(pos in left or pos in right for pos in _POSITIVE_TOKENS):
                has_positive = True
            if any(marker + signal in text for marker in question_markers):
                has_positive = True
                continue
            if any(f"{neg}{signal}" in text or f"{signal}{neg}" in text for neg in _NEGATION_TOKENS):
                return True

        if has_positive:
            return False

        # Generic fallback for expressions like "不用跑现场" against "线下看房".
        if "线下" in tag_norm and any(
            token in text for token in ("不用线下", "不想线下", "不跑现场", "不去现场", "不用跑线下", "不跑线下", "不去线下")
        ):
            return True
        if "线上" in tag_norm and any(token in text for token in ("希望线上", "想线上", "能线上", "可以线上")):
            return False
        if "线上" in tag_norm and any(token in text for token in ("不要线上", "不想线上")):
            return True
        return False

    @staticmethod
    def _is_must_intent_for_tag(text: str, tag: str, matched_signals: list[str]) -> bool:
        tag_norm = DialogueManager._normalize_tag_text(tag)
        if not tag_norm:
            return False
        if any(token in text for token in ("不一定", "不是必须", "非必须")):
            return False

        signals = matched_signals or [sig for sig in DialogueManager._extract_tag_signals(tag) if len(sig) >= 2]
        for signal in signals:
            if signal not in text:
                continue
            start = text.find(signal)
            if start >= 0:
                left = text[max(0, start - 8) : start]
                right = text[start + len(signal) : start + len(signal) + 8]
                if any(token in left or token in right for token in _MUST_TOKENS):
                    return True
            if any(f"{token}{signal}" in text or f"{signal}{token}" in text for token in _MUST_TOKENS):
                return True

        return False

    @staticmethod
    def _normalize_tag_text(text: str) -> str:
        lowered = text.lower()
        replaced = (
            lowered.replace("现场", "线下")
            .replace("实地", "线下")
            .replace("到店", "线下")
            .replace("在线", "线上")
            .replace("vr", "vr")
            .replace("ar", "ar")
        )
        return re.sub(r"[^0-9a-z\u4e00-\u9fa5]+", "", replaced)

    @staticmethod
    def _extract_tag_signals(tag: str) -> list[str]:
        normalized = DialogueManager._normalize_tag_text(tag)
        if not normalized:
            return []

        chunks = re.findall(r"[0-9a-z]+|[\u4e00-\u9fa5]{2,}", normalized)
        signals: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk in _TAG_TEXT_STOPWORDS:
                continue
            if len(chunk) >= 2 and chunk not in seen:
                seen.add(chunk)
                signals.append(chunk)
            if len(chunk) >= 3:
                for i in range(len(chunk) - 1):
                    part = chunk[i : i + 2]
                    if part in _TAG_TEXT_STOPWORDS or part in seen:
                        continue
                    seen.add(part)
                    signals.append(part)
        return signals

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
        state.req.hard = HardConstraints.model_validate(hard.model_dump())
        merged = DialogueManager._merge_tag_need_for_chat_memory(
            current=TagNeed(must=[], avoid=soft.avoid_tags, prefer=soft.preferred_tags),
            accumulated=state.req.soft.tag_need_accumulated,
            user_text=user_text,
        )
        state.req.soft.tag_need_accumulated = merged
        notes = list(state.req.soft.notes)
        for item in [*merged.must, *merged.avoid, *merged.prefer]:
            if item and item not in notes:
                notes.append(item)
        state.req.soft.notes = notes[-30:]

    @staticmethod
    def _merge_tag_need_for_chat_memory(*, current: TagNeed, accumulated: TagNeed, user_text: str) -> TagNeed:
        merged = TagNeed(
            must=list(dict.fromkeys([*accumulated.must, *current.must]))[:20],
            avoid=list(dict.fromkeys([*accumulated.avoid, *current.avoid]))[:20],
            prefer=list(dict.fromkeys([*accumulated.prefer, *current.prefer]))[:20],
        )
        return DialogueManager._apply_tag_need_revocations(merged, user_text)

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
                if field_name == "decoration" and isinstance(value, str):
                    normalized_decoration = _normalize_decoration_value(value)
                    if normalized_decoration is not None:
                        setattr(soft, field_name, normalized_decoration)
                        continue
                setattr(soft, field_name, value)

        merged_tag_need = TagNeed.model_validate(query.tag_need.model_dump())
        for item in soft.preferred_tags:
            if item and item not in merged_tag_need.prefer:
                merged_tag_need.prefer.append(item)
        for item in soft.avoid_tags:
            if item and item not in merged_tag_need.avoid:
                merged_tag_need.avoid.append(item)
        for item in state.req.soft.tag_need_accumulated.must:
            if item and item not in merged_tag_need.must:
                merged_tag_need.must.append(item)
        for item in state.req.soft.tag_need_accumulated.avoid:
            if item and item not in merged_tag_need.avoid:
                merged_tag_need.avoid.append(item)
        for item in state.req.soft.tag_need_accumulated.prefer:
            if item and item not in merged_tag_need.prefer:
                merged_tag_need.prefer.append(item)

        query.hard = hard
        query.soft = soft
        query.tag_need = TagNeed(
            must=merged_tag_need.must[:20],
            avoid=merged_tag_need.avoid[:20],
            prefer=merged_tag_need.prefer[:20],
        )
        return query

    def _apply_llm_parse(self, query: StructuredQuery, llm_parse: Any) -> StructuredQuery:
        if not isinstance(llm_parse, dict):
            return query

        parsed_intent = _normalize_intent(llm_parse.get("intent"))
        if parsed_intent is not None:
            query.intent = parsed_intent

        params_raw = llm_parse.get("params")
        if isinstance(params_raw, dict):
            overrides: dict[str, Any] = {}
            district = params_raw.get("district")
            if isinstance(district, str) and district.strip():
                overrides["district"] = district.strip()

            bedrooms = params_raw.get("bedrooms")
            if isinstance(bedrooms, str) and bedrooms.strip():
                compact = ",".join([item for item in re.findall(r"[1-9]", bedrooms) if item])
                if compact:
                    unique = []
                    seen = set()
                    for item in compact.split(","):
                        if item in seen:
                            continue
                        seen.add(item)
                        unique.append(item)
                    if unique:
                        layout = ",".join(unique)
                        overrides["layout"] = f"{layout}居"

            min_price = _to_int(params_raw.get("min_price"))
            if min_price is not None:
                overrides["budget_min"] = min_price
            max_price = _to_int(params_raw.get("max_price"))
            if max_price is not None:
                overrides["budget_max"] = max_price
            max_subway_dist = _to_int(params_raw.get("subway_distance"))
            if max_subway_dist is None:
                max_subway_dist = _to_int(params_raw.get("max_subway_dist"))
            if max_subway_dist is not None:
                overrides["max_subway_dist"] = max_subway_dist
            min_area = _to_float(params_raw.get("min_area"))
            if min_area is not None:
                overrides["area_min"] = min_area

            rental_type = params_raw.get("rental_type")
            if isinstance(rental_type, str) and rental_type.strip() in {"整租", "合租"}:
                overrides["rent_type"] = rental_type.strip()
            self._apply_hard_overrides(query.hard, overrides)

            soft_overrides: dict[str, Any] = {}
            decoration = params_raw.get("decoration")
            if isinstance(decoration, str) and decoration.strip():
                soft_overrides["decoration"] = decoration.strip()
            elevator = _to_bool(params_raw.get("elevator"))
            if elevator is not None:
                soft_overrides["elevator"] = elevator
            self._apply_soft_overrides(query.soft, soft_overrides)

        tag_need_raw = llm_parse.get("tag_need")
        if isinstance(tag_need_raw, dict):
            self._apply_tag_need_overrides(query.tag_need, tag_need_raw)

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
        known_landmarks = await self._get_known_landmark_aliases()
        normalized = cleaned[:-1] if cleaned.endswith("区") else cleaned
        if normalized in known_districts or cleaned in known_districts:
            query.hard.district = normalized
            return

        if cleaned in known_landmarks and cleaned not in known_districts:
            query.hard.area = cleaned
            query.hard.district = None
            if query.hard.landmark_name is None:
                query.hard.landmark_name = cleaned
            return

        if cleaned in user_text and not self._contains_explicit_admin_suffix_for(cleaned, user_text):
            query.hard.area = cleaned
            query.hard.district = None

    async def _get_known_district_aliases(self) -> set[str]:
        if self._known_district_aliases is not None:
            return self._known_district_aliases

        cached_aliases = getattr(self.cache, "landmark_district_aliases", None)
        if isinstance(cached_aliases, set) and cached_aliases:
            self._known_district_aliases = set(cached_aliases)
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

    async def _get_known_landmark_aliases(self) -> set[str]:
        if self._known_landmark_aliases is not None:
            return self._known_landmark_aliases

        cached_aliases = getattr(self.cache, "landmark_name_aliases", None)
        if isinstance(cached_aliases, set) and cached_aliases:
            self._known_landmark_aliases = set(cached_aliases)
            return self._known_landmark_aliases

        aliases: set[str] = set()
        client = self.landmarks_client
        list_fn = getattr(client, "list_landmarks", None) if client is not None else None
        if callable(list_fn):
            try:
                landmarks = await list_fn()
            except DataSourceError:
                landmarks = []
            for item in landmarks:
                name = getattr(item, "name", None)
                if not isinstance(name, str):
                    continue
                cleaned = name.strip()
                if not cleaned:
                    continue
                aliases.add(cleaned)
                if cleaned.endswith("站") and len(cleaned) > 1:
                    aliases.add(cleaned[:-1])

        self._known_landmark_aliases = aliases
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

            if key in {"amenities", "preferred_tags", "avoid_tags"}:
                if isinstance(value, list):
                    tokens = [str(item).strip() for item in value if str(item).strip()]
                elif isinstance(value, str):
                    tokens = [item.strip() for item in re.split(r"[,，/、\s]+", value) if item.strip()]
                else:
                    tokens = []
                if tokens:
                    current = getattr(soft, key)
                    setattr(soft, key, sorted(set(current + tokens)))
                continue

            if key in {"elevator", "value_for_money", "prioritize_subway_distance", "prefer_spacious", "prioritize_commute"}:
                parsed_bool = _to_bool(value)
                if parsed_bool is not None:
                    setattr(soft, key, parsed_bool)
                continue

            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    if key == "decoration":
                        normalized_decoration = _normalize_decoration_value(cleaned)
                        if normalized_decoration is not None:
                            soft.decoration = normalized_decoration
                        continue
                    setattr(soft, key, cleaned)

    @staticmethod
    def _apply_tag_need_overrides(tag_need: TagNeed, payload: dict[str, Any]) -> None:
        for key in ("must", "avoid", "prefer"):
            value = payload.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                tokens = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
            elif isinstance(value, str):
                tokens = [item.strip() for item in re.split(r"[,，/、\s]+", value) if item.strip()]
            else:
                tokens = []
            if not tokens:
                continue
            existing = getattr(tag_need, key)
            setattr(tag_need, key, sorted(set(existing + tokens)))

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
            if state.candidate_state.focus_house_id:
                return state.candidate_state.focus_house_id
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
        if state.candidate_state.latest_house_ids:
            return state.candidate_state.latest_house_ids
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
            state.candidate_state.latest_house_ids = house_ids
            state.candidate_state.focus_house_id = house_ids[0]
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
        bedroom_match = re.search(r"([一二两三四五六七八九123456789])\s*(?:居|室)", stripped)
        if bedroom_match:
            token = bedroom_match.group(1)
            if token.isdigit():
                return int(token)
            return _CH_NUM_TO_INT.get(token)

        compact = stripped.replace("，", "").replace(",", "")
        unit_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kK千wW万])", compact)
        if unit_match:
            base = float(unit_match.group(1))
            unit = unit_match.group(2).lower()
            if unit in {"k", "千"}:
                return int(base * 1000)
            if unit in {"w", "万"}:
                return int(base * 10000)

        cn_value = _coerce_cn_int(compact)
        if cn_value is not None:
            return cn_value
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


def _normalize_decoration_value(value: str) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned in {"精装", "精装修"}:
        return "精装"
    if cleaned in {"简装", "简装修"}:
        return "简装"
    return None


def _coerce_cn_int(text: str) -> int | None:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000}
    if not text or not re.fullmatch(r"[一二两三四五六七八九十百千万零〇]+", text):
        return None

    total = 0
    section = 0
    number = 0
    for ch in text:
        if ch in digits:
            number = digits[ch]
            continue
        if ch in units:
            unit = units[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
            continue
        if ch == "万":
            section += number
            if section == 0:
                section = 1
            total += section * 10000
            section = 0
            number = 0
            continue
        return None

    result = total + section + number
    return result if result > 0 else None
