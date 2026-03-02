from __future__ import annotations

from dataclasses import dataclass
import logging

from app.clients.exceptions import DataSourceError
from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.infra.cache import CacheManager
from app.infra.logging import log_event
from app.schemas import CaseType, HouseLite, StructuredQuery

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalPlan:
    plan_type: str
    landmark_id: str | None = None


class Planner:
    def __init__(
        self,
        landmarks_client: LandmarksClient,
        houses_client: HousesClient,
        *,
        cache: CacheManager | None = None,
        max_pages_single: int = 2,
        max_pages_multi: int = 3,
    ) -> None:
        self.landmarks_client = landmarks_client
        self.houses_client = houses_client
        self.cache = cache
        self.max_pages_single = max_pages_single
        self.max_pages_multi = max_pages_multi

    async def build_plan(self, query: StructuredQuery) -> RetrievalPlan:
        hard = query.hard
        if hard.landmark_name:
            landmark = None
            if self.cache is not None:
                landmark = self.cache.landmark_by_name.get(hard.landmark_name)
            if landmark is None:
                try:
                    landmark = await self.landmarks_client.get_by_name(hard.landmark_name)
                except DataSourceError:
                    log_event(LOGGER, "planner.landmark.lookup_failed", landmark_name=hard.landmark_name)
            if landmark is None:
                cands = await self.landmarks_client.search(hard.landmark_name, category=hard.landmark_category)
                if cands:
                    landmark = cands[0]
            if landmark is not None and self.cache is not None:
                self.cache.landmark_by_name[hard.landmark_name] = landmark
            if landmark and landmark.id:
                log_event(LOGGER, "planner.landmark.resolved", landmark_name=hard.landmark_name, landmark_id=landmark.id)
                return RetrievalPlan(plan_type="nearby_landmark", landmark_id=landmark.id)
        if hard.community:
            log_event(LOGGER, "planner.route.selected", route="by_community", community=hard.community)
            return RetrievalPlan(plan_type="by_community")
        log_event(LOGGER, "planner.route.selected", route="by_platform")
        return RetrievalPlan(plan_type="by_platform")

    async def execute_plan(self, plan: RetrievalPlan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
        max_pages = self.max_pages_single if case_type == CaseType.single else self.max_pages_multi
        if plan.plan_type == "nearby_landmark" and plan.landmark_id:
            return await self._fetch_nearby(plan.landmark_id, query, max_pages)
        if plan.plan_type == "by_community" and query.hard.community:
            return await self._fetch_by_community(query, max_pages)
        return await self._fetch_by_platform(query, max_pages)

    async def _fetch_nearby(self, landmark_id: str, query: StructuredQuery, max_pages: int) -> list[HouseLite]:
        merged: dict[str, HouseLite] = {}
        for page in range(1, max_pages + 1):
            log_event(LOGGER, "planner.fetch.start", source="nearby", page=page, landmark_id=landmark_id)
            resp = await self.houses_client.nearby(
                landmark_id=landmark_id,
                max_distance=2000,
                listing_platform=query.hard.listing_platform.value if query.hard.listing_platform else None,
                page=page,
                page_size=10,
            )
            log_event(LOGGER, "planner.fetch.done", source="nearby", page=page, item_count=len(resp["items"]))
            for house in resp["items"]:
                merged[house.house_id] = house
            if not resp["items"]:
                break
        return list(merged.values())

    async def _fetch_by_community(self, query: StructuredQuery, max_pages: int) -> list[HouseLite]:
        merged: dict[str, HouseLite] = {}
        for page in range(1, max_pages + 1):
            log_event(LOGGER, "planner.fetch.start", source="by_community", page=page, community=query.hard.community)
            resp = await self.houses_client.by_community(
                community=query.hard.community or "",
                listing_platform=query.hard.listing_platform.value if query.hard.listing_platform else None,
                page=page,
                page_size=10,
            )
            log_event(LOGGER, "planner.fetch.done", source="by_community", page=page, item_count=len(resp["items"]))
            for house in resp["items"]:
                merged[house.house_id] = house
            if not resp["items"]:
                break
        return list(merged.values())

    async def _fetch_by_platform(self, query: StructuredQuery, max_pages: int) -> list[HouseLite]:
        merged: dict[str, HouseLite] = {}
        for page in range(1, max_pages + 1):
            log_event(
                LOGGER,
                "planner.fetch.start",
                source="by_platform",
                page=page,
                district=query.hard.district,
                area=query.hard.area,
            )
            resp = await self.houses_client.by_platform(
                listing_platform=query.hard.listing_platform.value if query.hard.listing_platform else None,
                page=page,
                page_size=10,
                district=query.hard.district,
                area=query.hard.area,
                min_price=query.hard.budget_min,
                max_price=query.hard.budget_max,
                min_area=int(query.hard.area_min) if query.hard.area_min is not None else None,
                commute_to_xierqi_max=query.hard.max_commute_min,
                max_subway_dist=query.hard.max_subway_dist,
                rental_type=query.hard.rent_type,
                available_from_before=query.hard.move_in_date,
            )
            log_event(LOGGER, "planner.fetch.done", source="by_platform", page=page, item_count=len(resp["items"]))
            for house in resp["items"]:
                merged[house.house_id] = house
            if not resp["items"]:
                break
        return list(merged.values())
