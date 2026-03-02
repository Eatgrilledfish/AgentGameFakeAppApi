import asyncio

from app.agent.planner import Planner
from app.clients.exceptions import DataSourceError
from app.schemas import HardConstraints, Landmark, StructuredQuery


class DummyLandmarksClient:
    def __init__(self) -> None:
        self.search_called = False

    async def get_by_name(self, name: str):
        raise DataSourceError("404")

    async def search(self, keyword: str, category=None, district=None):
        self.search_called = True
        return [Landmark(id="SS_001", name="车公庄站", category="subway")]


class DummyHousesClient:
    pass


def test_planner_falls_back_to_search_when_name_lookup_fails() -> None:
    landmarks = DummyLandmarksClient()
    planner = Planner(landmarks_client=landmarks, houses_client=DummyHousesClient())
    query = StructuredQuery(hard=HardConstraints(landmark_name="车公庄"))

    plan = asyncio.run(planner.build_plan(query))

    assert landmarks.search_called is True
    assert plan.plan_type == "nearby_landmark"
    assert plan.landmark_id == "SS_001"
