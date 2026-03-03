import pytest

from app.agent.ranker import Ranker
from app.schemas import HardConstraints, HouseLite, Listing, SoftPreferences, StructuredQuery
from app.settings import RankingWeights


class DummyHousesClient:
    def __init__(self) -> None:
        self.listings_calls = 0

    async def get_listings(self, house_id: str) -> dict:
        self.listings_calls += 1
        return {
            "items": [
                Listing(listing_platform="安居客", rent=6000, status="可租"),
                Listing(listing_platform="链家", rent=6200, status="可租"),
            ]
        }

    async def nearby_landmarks(self, community: str, category: str, max_distance_m: int = 3000):
        return []


@pytest.mark.asyncio
async def test_ranker_returns_top_candidates() -> None:
    ranker = Ranker(houses_client=DummyHousesClient(), weights=RankingWeights(), enrich_concurrency=2)
    query = StructuredQuery(
        hard=HardConstraints(budget_max=8000, max_subway_dist=1000, max_commute_min=40),
        soft=SoftPreferences(),
    )
    candidates = [
        HouseLite(
            house_id="HF_1",
            rent=6000,
            area=70,
            district="海淀",
            community="西二旗家园",
            subway_distance=500,
            commute_to_xierqi_min=20,
            status="可租",
            layout="2居1厅1卫",
            tags=["近地铁"],
            decoration="精装",
            elevator=True,
            orientation="朝南",
        ),
        HouseLite(
            house_id="HF_2",
            rent=7800,
            area=65,
            district="海淀",
            community="上地佳园",
            subway_distance=1200,
            commute_to_xierqi_min=35,
            status="可租",
            layout="2居1厅1卫",
            tags=[],
        ),
    ]

    top = await ranker.rank_two_stage(candidates, query, max_output=5)

    assert len(top) >= 1
    assert top[0].house_id == "HF_1"


@pytest.mark.asyncio
async def test_ranker_relaxes_budget_by_ten_percent_when_no_results() -> None:
    ranker = Ranker(houses_client=DummyHousesClient(), weights=RankingWeights(), enrich_concurrency=2)
    query = StructuredQuery(
        hard=HardConstraints(budget_max=8000, max_subway_dist=1000, max_commute_min=40),
        soft=SoftPreferences(),
    )
    candidates = [
        HouseLite(
            house_id="HF_3",
            rent=8500,
            area=70,
            district="海淀",
            community="西二旗家园",
            subway_distance=500,
            commute_to_xierqi_min=20,
            status="可租",
            layout="2居1厅1卫",
            tags=[],
        )
    ]

    top = await ranker.rank_two_stage(candidates, query, max_output=5)

    assert len(top) == 1
    assert any("放宽预算上限 10%" in c for c in top[0].cons)


@pytest.mark.asyncio
async def test_ranker_accepts_english_available_status() -> None:
    ranker = Ranker(houses_client=DummyHousesClient(), weights=RankingWeights(), enrich_concurrency=2)
    query = StructuredQuery(
        hard=HardConstraints(budget_max=8000, max_subway_dist=1000, max_commute_min=40),
        soft=SoftPreferences(),
    )
    candidates = [
        HouseLite(
            house_id="HF_4",
            rent=7000,
            area=60,
            district="海淀",
            community="西二旗家园",
            subway_distance=400,
            commute_to_xierqi_min=18,
            status="available",
            layout="2居1厅1卫",
            tags=["近地铁"],
        )
    ]

    top = await ranker.rank_two_stage(candidates, query, max_output=5)

    assert len(top) == 1
    assert top[0].house_id == "HF_4"


@pytest.mark.asyncio
async def test_ranker_prioritizes_subway_distance_when_requested() -> None:
    ranker = Ranker(houses_client=DummyHousesClient(), weights=RankingWeights(), enrich_concurrency=2)
    query = StructuredQuery(
        hard=HardConstraints(max_subway_dist=800),
        soft=SoftPreferences(prioritize_subway_distance=True),
    )
    candidates = [
        HouseLite(
            house_id="HF_FAR",
            rent=5000,
            area=70,
            district="西城",
            community="A",
            subway_distance=700,
            commute_to_xierqi_min=20,
            status="available",
            layout="1居1厅1卫",
            tags=[],
        ),
        HouseLite(
            house_id="HF_NEAR",
            rent=9000,
            area=50,
            district="西城",
            community="B",
            subway_distance=200,
            commute_to_xierqi_min=35,
            status="available",
            layout="1居1厅1卫",
            tags=[],
        ),
    ]

    top = await ranker.rank_two_stage(candidates, query, max_output=2)

    assert [item.house_id for item in top] == ["HF_NEAR", "HF_FAR"]


@pytest.mark.asyncio
async def test_ranker_skips_listings_enrichment_by_default() -> None:
    client = DummyHousesClient()
    ranker = Ranker(houses_client=client, weights=RankingWeights(), enrich_concurrency=2)
    query = StructuredQuery(
        hard=HardConstraints(budget_max=8000, max_subway_dist=1000),
        soft=SoftPreferences(),
    )
    candidates = [
        HouseLite(
            house_id="HF_1",
            rent=6000,
            area=70,
            district="海淀",
            community="西二旗家园",
            subway_distance=500,
            commute_to_xierqi_min=20,
            status="available",
            layout="2居1厅1卫",
            tags=["近地铁"],
        )
    ]

    await ranker.rank_two_stage(candidates, query, max_output=1)

    assert client.listings_calls == 0


@pytest.mark.asyncio
async def test_ranker_boosts_candidates_with_tags_matching_user_preferences() -> None:
    ranker = Ranker(houses_client=DummyHousesClient(), weights=RankingWeights(), enrich_concurrency=2)
    query = StructuredQuery(
        hard=HardConstraints(budget_max=7000),
        soft=SoftPreferences(
            elevator=True,
            orientation="朝南",
            value_for_money=True,
            amenities=["公园"],
        ),
    )
    candidates = [
        HouseLite(
            house_id="HF_MATCH",
            rent=6200,
            area=68,
            district="朝阳",
            community="A",
            subway_distance=650,
            commute_to_xierqi_min=36,
            status="available",
            layout="2居1厅1卫",
            tags=["有电梯", "朝南", "高性价比", "近公园"],
        ),
        HouseLite(
            house_id="HF_PLAIN",
            rent=6200,
            area=68,
            district="朝阳",
            community="B",
            subway_distance=650,
            commute_to_xierqi_min=36,
            status="available",
            layout="2居1厅1卫",
            tags=["普通装修"],
        ),
    ]

    top = await ranker.rank_two_stage(candidates, query, max_output=2)

    assert [item.house_id for item in top] == ["HF_MATCH", "HF_PLAIN"]


@pytest.mark.asyncio
async def test_ranker_penalizes_avoid_tags_and_prioritizes_preferred_tags() -> None:
    ranker = Ranker(houses_client=DummyHousesClient(), weights=RankingWeights(), enrich_concurrency=2)
    query = StructuredQuery(
        hard=HardConstraints(budget_max=7000),
        soft=SoftPreferences(
            preferred_tags=["采光好", "近公园"],
            avoid_tags=["收中介费", "临街"],
        ),
    )
    candidates = [
        HouseLite(
            house_id="HF_SAFE",
            rent=6200,
            area=68,
            district="朝阳",
            community="A",
            subway_distance=650,
            commute_to_xierqi_min=36,
            status="available",
            layout="2居1厅1卫",
            tags=["采光好", "近公园", "房东直租"],
        ),
        HouseLite(
            house_id="HF_RISKY",
            rent=6200,
            area=68,
            district="朝阳",
            community="B",
            subway_distance=650,
            commute_to_xierqi_min=36,
            status="available",
            layout="2居1厅1卫",
            tags=["采光好", "近公园", "收中介费", "临街"],
        ),
    ]

    top = await ranker.rank_two_stage(candidates, query, max_output=2)

    assert [item.house_id for item in top] == ["HF_SAFE", "HF_RISKY"]
