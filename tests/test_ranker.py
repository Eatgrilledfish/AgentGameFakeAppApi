import pytest

from app.agent.ranker import Ranker
from app.schemas import HardConstraints, HouseLite, Listing, SoftPreferences, StructuredQuery
from app.settings import RankingWeights


class DummyHousesClient:
    async def get_listings(self, house_id: str) -> dict:
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
