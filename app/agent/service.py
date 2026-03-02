from __future__ import annotations

import logging
import math

from app.agent.budget import BudgetManager
from app.agent.dialogue import DialogueManager
from app.agent.formatter import OutputFormatter
from app.agent.nlu import RuleBasedNLU
from app.agent.planner import Planner
from app.agent.ranker import Ranker
from app.agent.state import StateStore
from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.infra.cache import CacheManager
from app.infra.logging import log_event
from app.schemas import InvokeRequest, InvokeResponse
from app.settings import AgentSettings

LOGGER = logging.getLogger(__name__)


class AgentService:
    def __init__(
        self,
        *,
        settings: AgentSettings,
        state_store: StateStore,
        landmarks_client: LandmarksClient,
        houses_client: HousesClient,
        cache: CacheManager,
    ) -> None:
        self.settings = settings
        self.state_store = state_store
        self.landmarks_client = landmarks_client
        self.houses_client = houses_client
        self.cache = cache
        self.budget = BudgetManager()

        self.dialogue = DialogueManager(
            state_store=state_store,
            nlu=RuleBasedNLU(),
            planner=Planner(
                landmarks_client=landmarks_client,
                houses_client=houses_client,
                cache=cache,
                max_pages_single=settings.max_pages_single,
                max_pages_multi=settings.max_pages_multi,
            ),
            ranker=Ranker(
                houses_client=houses_client,
                weights=settings.weights,
                enrich_concurrency=settings.enrich_concurrency,
                cache=cache,
                listing_top_n=settings.enrich_listing_top_n,
                amenities_top_n=settings.enrich_amenities_top_n,
            ),
            formatter=OutputFormatter(),
            houses_client=houses_client,
            cache=cache,
            max_output_candidates=settings.max_output_candidates,
        )

    async def handle(self, request: InvokeRequest) -> InvokeResponse:
        user_id = request.user_id or self.settings.default_user_id
        log_event(
            LOGGER,
            "agent.handle.start",
            session_id=request.session_id,
            case_type=request.case_type.value,
            user_id=user_id,
        )
        state, is_new = self.state_store.get_or_create(request.session_id, user_id, request.case_type)
        log_event(
            LOGGER,
            "agent.state.ready",
            is_new_session=is_new,
            phase=state.phase.value,
            used_slices=state.budget.used_slices,
            used_tokens=state.budget.used_tokens,
        )
        if state.budget.limit_slices != self.settings.budget_limit_slices:
            state.budget.limit_slices = self.settings.budget_limit_slices
        resp = await self.dialogue.handle_turn(request, state, is_new_session=is_new)
        log_event(
            LOGGER,
            "agent.handle.done",
            phase=state.phase.value,
            candidate_count=len(resp.candidates),
            clarify_count=len(resp.clarify_questions),
        )
        return resp

    def allow_llm_fallback(self, session_id: str, estimated_prompt_tokens: int) -> bool:
        state = self.state_store.get(session_id)
        if state is None:
            return True
        return self.budget.can_use_llm(state.budget, estimated_prompt_tokens)

    def record_llm_fallback_usage(self, session_id: str, consumed_tokens: int) -> None:
        state = self.state_store.get(session_id)
        if state is None:
            return
        self.budget.record_llm_usage(state.budget, consumed_tokens)
        self.state_store.upsert(state)

    @staticmethod
    def rough_token_estimate(text: str) -> int:
        # Lightweight heuristic: Chinese chars and ASCII mix roughly map to 1 token per ~1.8 chars.
        return max(1, int(math.ceil(len(text) / 1.8)))
