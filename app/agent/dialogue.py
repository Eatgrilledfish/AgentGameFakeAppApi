from __future__ import annotations

import logging

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
    IntentType,
    InvokeRequest,
    InvokeResponse,
    SessionPhase,
    SessionState,
    SoftPreferences,
    StructuredQuery,
)

LOGGER = logging.getLogger(__name__)


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

        query = self.nlu.parse(request.message, state, request.case_type)
        merged = self._merge_query_with_state(query, state)
        log_event(
            LOGGER,
            "dialogue.nlu.done",
            intent=merged.intent.value,
            hard=merged.hard.model_dump(exclude_none=True),
            soft=merged.soft.model_dump(exclude_none=True),
            clarify_count=len(merged.clarify_questions),
        )

        if merged.intent in {IntentType.rent, IntentType.terminate, IntentType.offline}:
            log_event(LOGGER, "dialogue.action.detected", intent=merged.intent.value)
            resp = await self._handle_action(merged, state)
            self.state_store.upsert(state)
            return resp

        if request.case_type.value == "Multi" and merged.clarify_questions and state.phase in {
            SessionPhase.chatting,
            SessionPhase.slot_filling,
        }:
            log_event(LOGGER, "dialogue.clarify", questions=merged.clarify_questions)
            state.phase = SessionPhase.slot_filling
            self.state_store.upsert(state)
            return self.formatter.render(
                case_type=request.case_type,
                query=merged,
                top_houses=[],
                clarify_questions=merged.clarify_questions,
            )

        if merged.intent == IntentType.chat:
            log_event(LOGGER, "dialogue.chat")
            state.phase = SessionPhase.chatting
            self.state_store.upsert(state)
            return InvokeResponse(text="你好，我可以按预算、地铁距离、通勤和小区等条件帮你找房。")

        try:
            state.phase = SessionPhase.searching
            plan = await self.planner.build_plan(merged)
            log_event(LOGGER, "dialogue.plan.built", plan_type=plan.plan_type, landmark_id=plan.landmark_id)
            candidates = await self.planner.execute_plan(plan, merged, request.case_type)
            log_event(LOGGER, "dialogue.plan.executed", candidate_count=len(candidates))
            top = await self.ranker.rank_two_stage(candidates, merged, max_output=self.max_output_candidates)
            log_event(LOGGER, "dialogue.rank.done", top_count=len(top))

            state.phase = SessionPhase.presenting
            state.confirmed_constraints = merged.hard
            state.soft_preferences = merged.soft
            state.last_candidates = candidates
            state.last_top5 = top
            self.state_store.upsert(state)
            return self.formatter.render(
                case_type=request.case_type,
                query=merged,
                top_houses=top,
                debug={"plan": plan.plan_type, "candidate_count": len(candidates)},
            )
        except DataSourceError as exc:
            log_event(LOGGER, "dialogue.error", error=str(exc))
            LOGGER.exception("Data source error")
            return InvokeResponse(text=f"检索接口调用失败：{exc}，请稍后重试。", candidates=[])

    async def _init_session_data(self, state: SessionState) -> None:
        try:
            log_event(LOGGER, "dialogue.session.init.start")
            await self.houses_client.init_houses()
            self.cache.invalidate_all_houses()
            log_event(LOGGER, "dialogue.session.init.done")
        except DataSourceError:
            LOGGER.warning("init houses failed for session=%s", state.session_id)
            log_event(LOGGER, "dialogue.session.init.failed")

    async def _handle_action(self, query: StructuredQuery, state: SessionState) -> InvokeResponse:
        if not query.hard.house_id or not query.hard.listing_platform:
            questions = []
            if not query.hard.house_id:
                questions.append("请补充 house_id，例如 HF_2001。")
            if not query.hard.listing_platform:
                questions.append("请补充挂牌平台（链家/安居客/58同城）。")
            return self.formatter.render(
                case_type=state.case_type,
                query=query,
                top_houses=[],
                clarify_questions=questions,
            )

        state.phase = SessionPhase.executing
        house_id = query.hard.house_id
        platform = query.hard.listing_platform.value
        log_event(
            LOGGER,
            "dialogue.action.start",
            intent=query.intent.value,
            house_id=house_id,
            listing_platform=platform,
        )

        if query.intent == IntentType.rent:
            result = await self.houses_client.rent(house_id, platform)
            action = "rent"
        elif query.intent == IntentType.terminate:
            result = await self.houses_client.terminate(house_id, platform)
            action = "terminate"
        else:
            result = await self.houses_client.offline(house_id, platform)
            action = "offline"

        self.cache.invalidate_house(house_id)
        self.cache.invalidate_query_cache()
        state.phase = SessionPhase.presenting
        log_event(LOGGER, "dialogue.action.done", action=action, result=result)
        return self.formatter.render_action_result(action, result)

    def _merge_query_with_state(self, query: StructuredQuery, state: SessionState) -> StructuredQuery:
        hard = HardConstraints.model_validate(state.confirmed_constraints.model_dump())
        soft = SoftPreferences.model_validate(state.soft_preferences.model_dump())

        for field_name, value in query.hard.model_dump().items():
            if value is not None:
                setattr(hard, field_name, value)

        for field_name, value in query.soft.model_dump().items():
            if isinstance(value, list) and value:
                merged = sorted(set(getattr(soft, field_name) + value))
                setattr(soft, field_name, merged)
            elif value is not None:
                setattr(soft, field_name, value)

        query.hard = hard
        query.soft = soft
        return query
