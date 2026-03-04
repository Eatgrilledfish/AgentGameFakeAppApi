import asyncio

from app.agent.planner import Planner, RetrievalPlan
from app.clients.exceptions import DataSourceError
from app.schemas import CaseType, HardConstraints, HouseLite, Landmark, StructuredQuery


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


class NearbyCaptureHousesClient:
    def __init__(self) -> None:
        self.last_max_distance = None

    async def nearby(self, landmark_id: str, max_distance=2000, listing_platform=None, page=1, page_size=10):
        self.last_max_distance = max_distance
        return {"items": [], "total": 0, "page_size": 10}


class FuzzyLandmarksClient:
    async def get_by_name(self, name: str):
        return None

    async def search(self, keyword: str, category=None, district=None):
        if keyword == "车公庄站":
            return [Landmark(id="SS_001", name="车公庄站", category="subway")]
        return []


class NearbyWithFallbackHousesClient:
    def __init__(self) -> None:
        self.nearby_calls: list[str] = []
        self.by_platform_calls = 0
        self.by_community_calls = 0

    async def nearby(self, landmark_id: str, max_distance=2000, listing_platform=None, page=1, page_size=10):
        self.nearby_calls.append(landmark_id)
        if landmark_id == "SS_001":
            return {
                "items": [HouseLite(house_id="HF_NEAR_1", district="西城", layout="2居1厅1卫", rent=4300, status="可租")],
                "total": 1,
                "page_size": 10,
            }
        return {"items": [], "total": 0, "page_size": 10}

    async def by_platform(self, **kwargs):
        self.by_platform_calls += 1
        return {
            "items": [HouseLite(house_id="HF_PLATFORM_1", district="西城", layout="2居1厅1卫", rent=4500, status="可租")],
            "total": 1,
            "page_size": 10,
        }

    async def by_community(self, **kwargs):
        self.by_community_calls += 1
        return {"items": [], "total": 0, "page_size": 10}


class ByPlatformCaptureHousesClient:
    def __init__(self) -> None:
        self.by_platform_calls: list[dict] = []

    async def by_platform(self, **kwargs):
        self.by_platform_calls.append(kwargs)
        page = int(kwargs.get("page", 1))
        if page == 1:
            return {
                "items": [HouseLite(house_id="HF_PLATFORM_O1", district="朝阳", layout="2居1厅1卫", rent=5600, status="可租")],
                "total": 1,
                "page_size": 10,
            }
        return {"items": [], "total": 0, "page_size": 10}


def test_planner_falls_back_to_search_when_name_lookup_fails() -> None:
    landmarks = DummyLandmarksClient()
    planner = Planner(landmarks_client=landmarks, houses_client=DummyHousesClient())
    query = StructuredQuery(hard=HardConstraints(landmark_name="车公庄"))

    plan = asyncio.run(planner.build_plan(query))

    assert landmarks.search_called is True
    assert plan.plan_type == "nearby_landmark"
    assert plan.landmark_id == "SS_001"


def test_planner_nearby_uses_llm_max_distance_when_provided() -> None:
    landmarks = DummyLandmarksClient()
    houses = NearbyCaptureHousesClient()
    planner = Planner(landmarks_client=landmarks, houses_client=houses)
    query = StructuredQuery(hard=HardConstraints(landmark_id="SS_001", max_distance=500))
    plan = RetrievalPlan(plan_type="nearby_landmark", landmark_id="SS_001")

    asyncio.run(planner.execute_plan(plan, query, CaseType.single))

    assert houses.last_max_distance == 500


def test_planner_nearby_degrades_to_fuzzy_landmark_search_then_hits() -> None:
    landmarks = FuzzyLandmarksClient()
    houses = NearbyWithFallbackHousesClient()
    planner = Planner(landmarks_client=landmarks, houses_client=houses)
    query = StructuredQuery(hard=HardConstraints(landmark_id="车公庄站", max_distance=500))
    plan = RetrievalPlan(plan_type="nearby_landmark", landmark_id="车公庄站")

    rows = asyncio.run(planner.execute_plan(plan, query, CaseType.single))

    assert houses.nearby_calls[:2] == ["车公庄站", "SS_001"]
    assert len(rows) == 1
    assert rows[0].house_id == "HF_NEAR_1"
    assert query.hard.landmark_id == "SS_001"


def test_planner_nearby_degrades_to_by_platform_when_still_empty() -> None:
    landmarks = DummyLandmarksClient()
    houses = NearbyCaptureHousesClient()

    async def by_platform(**kwargs):
        return {
            "items": [HouseLite(house_id="HF_FALLBACK_1", district="海淀", layout="2居1厅1卫", rent=5200, status="可租")],
            "total": 1,
            "page_size": 10,
        }

    houses.by_platform = by_platform  # type: ignore[attr-defined]
    planner = Planner(landmarks_client=landmarks, houses_client=houses)
    query = StructuredQuery(hard=HardConstraints(landmark_id="unknown_landmark"))
    plan = RetrievalPlan(plan_type="nearby_landmark", landmark_id="unknown_landmark")

    rows = asyncio.run(planner.execute_plan(plan, query, CaseType.single))

    assert len(rows) == 1
    assert rows[0].house_id == "HF_FALLBACK_1"


def test_planner_by_community_degrades_to_by_platform_when_empty() -> None:
    landmarks = FuzzyLandmarksClient()
    houses = NearbyWithFallbackHousesClient()
    planner = Planner(landmarks_client=landmarks, houses_client=houses)
    query = StructuredQuery(hard=HardConstraints(community="不存在小区"))
    plan = RetrievalPlan(plan_type="by_community")

    rows = asyncio.run(planner.execute_plan(plan, query, CaseType.single))

    assert houses.by_community_calls == 1
    assert houses.by_platform_calls >= 1
    assert len(rows) == 1
    assert rows[0].house_id == "HF_PLATFORM_1"


def test_planner_by_platform_passes_soft_preferences_to_upstream_params() -> None:
    landmarks = FuzzyLandmarksClient()
    houses = ByPlatformCaptureHousesClient()
    planner = Planner(landmarks_client=landmarks, houses_client=houses)
    query = StructuredQuery(
        hard=HardConstraints(area_min=60, max_subway_dist=800),
        soft={"orientation": "朝南", "elevator": True, "decoration": "精装修"},
    )
    plan = RetrievalPlan(plan_type="by_platform")

    rows = asyncio.run(planner.execute_plan(plan, query, CaseType.single))

    assert len(rows) == 1
    assert rows[0].house_id == "HF_PLATFORM_O1"
    assert houses.by_platform_calls
    first = houses.by_platform_calls[0]
    assert first["min_area"] == 60
    assert first["subway_distance"] == 800
    assert first["orientation"] == "朝南"
    assert first["elevator"] == "true"
    assert first["decoration"] == "精装"


def test_planner_layout_range_maps_to_bedrooms_csv() -> None:
    landmarks = FuzzyLandmarksClient()
    houses = ByPlatformCaptureHousesClient()
    planner = Planner(landmarks_client=landmarks, houses_client=houses)
    query = StructuredQuery(
        hard=HardConstraints(layout="2,3居"),
        soft={},
    )
    plan = RetrievalPlan(plan_type="by_platform")

    _ = asyncio.run(planner.execute_plan(plan, query, CaseType.single))

    assert houses.by_platform_calls
    first = houses.by_platform_calls[0]
    assert first["bedrooms"] == "2,3"
