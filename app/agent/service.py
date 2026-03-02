from __future__ import annotations

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
from app.schemas import InvokeRequest, InvokeResponse
from app.settings import AgentSettings


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
            planner=Planner(landmarks_client=landmarks_client, houses_client=houses_client),
            ranker=Ranker(
                houses_client=houses_client,
                weights=settings.weights,
                enrich_concurrency=settings.enrich_concurrency,
            ),
            formatter=OutputFormatter(),
            houses_client=houses_client,
            cache=cache,
            max_output_candidates=settings.max_output_candidates,
        )

    async def handle(self, request: InvokeRequest) -> InvokeResponse:
        user_id = request.user_id or self.settings.default_user_id
        state, is_new = self.state_store.get_or_create(request.session_id, user_id, request.case_type)
        if state.budget.limit_slices != self.settings.budget_limit_slices:
            state.budget.limit_slices = self.settings.budget_limit_slices
        return await self.dialogue.handle_turn(request, state, is_new_session=is_new)
