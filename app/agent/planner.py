from __future__ import annotations

from dataclasses import dataclass

from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.schemas import CaseType, HouseLite, StructuredQuery


@dataclass(slots=True)
class RetrievalPlan:
    plan_type: str
    landmark_id: str | None = None


class Planner:
    def __init__(self, landmarks_client: LandmarksClient, houses_client: HousesClient) -> None:
        self.landmarks_client = landmarks_client
        self.houses_client = houses_client

    async def build_plan(self, query: StructuredQuery) -> RetrievalPlan:
        hard = query.hard
        if hard.landmark_name:
            landmark = await self.landmarks_client.get_by_name(hard.landmark_name)
            if landmark is None:
                cands = await self.landmarks_client.search(hard.landmark_name, category=hard.landmark_category)
                if cands:
                    landmark = cands[0]
            if landmark and landmark.id:
                return RetrievalPlan(plan_type="nearby_landmark", landmark_id=landmark.id)
        if hard.community:
            return RetrievalPlan(plan_type="by_community")
        return RetrievalPlan(plan_type="by_platform")

    async def execute_plan(self, plan: RetrievalPlan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
        max_pages = 2 if case_type == CaseType.single else 3
        if plan.plan_type == "nearby_landmark" and plan.landmark_id:
            return await self._fetch_nearby(plan.landmark_id, query, max_pages)
        if plan.plan_type == "by_community" and query.hard.community:
            return await self._fetch_by_community(query, max_pages)
        return await self._fetch_by_platform(query, max_pages)

    async def _fetch_nearby(self, landmark_id: str, query: StructuredQuery, max_pages: int) -> list[HouseLite]:
        merged: dict[str, HouseLite] = {}
        for page in range(1, max_pages + 1):
            resp = await self.houses_client.nearby(
                landmark_id=landmark_id,
                max_distance=2000,
                listing_platform=query.hard.listing_platform.value if query.hard.listing_platform else None,
                page=page,
                page_size=10,
                max_subway_dist=query.hard.max_subway_dist,
            )
            for house in resp["items"]:
                merged[house.house_id] = house
            if not resp["items"]:
                break
        return list(merged.values())

    async def _fetch_by_community(self, query: StructuredQuery, max_pages: int) -> list[HouseLite]:
        merged: dict[str, HouseLite] = {}
        for page in range(1, max_pages + 1):
            resp = await self.houses_client.by_community(
                community=query.hard.community or "",
                listing_platform=query.hard.listing_platform.value if query.hard.listing_platform else None,
                page=page,
                page_size=10,
                max_subway_dist=query.hard.max_subway_dist,
            )
            for house in resp["items"]:
                merged[house.house_id] = house
            if not resp["items"]:
                break
        return list(merged.values())

    async def _fetch_by_platform(self, query: StructuredQuery, max_pages: int) -> list[HouseLite]:
        merged: dict[str, HouseLite] = {}
        for page in range(1, max_pages + 1):
            resp = await self.houses_client.by_platform(
                listing_platform=query.hard.listing_platform.value if query.hard.listing_platform else None,
                page=page,
                page_size=10,
                district=query.hard.district,
                min_price=query.hard.budget_min,
                max_price=query.hard.budget_max,
                min_area=query.hard.area_min,
                commute_to_xierqi_max=query.hard.max_commute_min,
                max_subway_dist=query.hard.max_subway_dist,
                rental_type=query.hard.rent_type,
            )
            for house in resp["items"]:
                merged[house.house_id] = house
            if not resp["items"]:
                break
        return list(merged.values())
