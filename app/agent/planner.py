from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

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
        if hard.landmark_id:
            log_event(LOGGER, "planner.landmark.direct_id", landmark_id=hard.landmark_id)
            return RetrievalPlan(plan_type="nearby_landmark", landmark_id=hard.landmark_id)
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
            primary = await self._fetch_nearby(plan.landmark_id, query, max_pages)
            if primary:
                return primary

            degraded_nearby = await self._degrade_nearby_via_fuzzy_landmark(
                original_landmark=plan.landmark_id,
                query=query,
                max_pages=max_pages,
            )
            if degraded_nearby:
                return degraded_nearby

            return await self._degrade_to_by_platform(
                query=query,
                max_pages=max_pages,
                from_route="nearby_landmark",
                reason="nearby_empty_after_fuzzy",
            )
        if plan.plan_type == "by_community" and query.hard.community:
            primary = await self._fetch_by_community(query, max_pages)
            if primary:
                return primary
            return await self._degrade_to_by_platform(
                query=query,
                max_pages=max_pages,
                from_route="by_community",
                reason="by_community_empty",
            )
        return await self._fetch_by_platform(query, max_pages)

    async def _fetch_nearby(self, landmark_id: str, query: StructuredQuery, max_pages: int) -> list[HouseLite]:
        merged: dict[str, HouseLite] = {}
        max_distance = (
            query.hard.max_distance
            if query.hard.max_distance is not None
            else (query.hard.max_subway_dist if query.hard.max_subway_dist is not None else 2000)
        )
        for page in range(1, max_pages + 1):
            log_event(LOGGER, "planner.fetch.start", source="nearby", page=page, landmark_id=landmark_id)
            resp = await self.houses_client.nearby(
                landmark_id=landmark_id,
                max_distance=max_distance,
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
                bedrooms=_layout_to_bedrooms(query.hard.layout),
                min_area=int(query.hard.area_min) if query.hard.area_min is not None else None,
                orientation=query.soft.orientation,
                decoration=_normalize_decoration_param(query.soft.decoration),
                elevator=_to_elevator_param(query.soft.elevator),
                commute_to_xierqi_max=query.hard.max_commute_min,
                utilities_type=query.hard.utilities_type,
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

    async def _degrade_nearby_via_fuzzy_landmark(
        self,
        *,
        original_landmark: str,
        query: StructuredQuery,
        max_pages: int,
    ) -> list[HouseLite]:
        search = getattr(self.landmarks_client, "search", None)
        if not callable(search):
            return []

        keywords = _landmark_fuzzy_keywords(
            original_landmark=original_landmark,
            landmark_name=query.hard.landmark_name,
        )
        if not keywords:
            return []

        attempted_landmark_ids = {original_landmark}
        for keyword in keywords:
            try:
                candidates = await search(
                    keyword,
                    category=query.hard.landmark_category,
                    district=query.hard.district,
                )
            except DataSourceError:
                log_event(LOGGER, "planner.degrade.landmark.search_failed", keyword=keyword)
                continue

            if not candidates:
                log_event(LOGGER, "planner.degrade.landmark.search_empty", keyword=keyword)
                continue

            picked = next(
                (item for item in candidates if _has_landmark_id(item) and str(item.id) not in attempted_landmark_ids),
                None,
            )
            if picked is None:
                continue

            picked_id = str(picked.id)
            attempted_landmark_ids.add(picked_id)
            if self.cache is not None:
                self.cache.landmark_by_name[keyword] = picked

            query.hard.landmark_id = picked_id
            if not query.hard.landmark_name and _has_landmark_name(picked):
                query.hard.landmark_name = str(picked.name)

            log_event(
                LOGGER,
                "planner.degrade.landmark.retry_nearby",
                source_landmark=original_landmark,
                keyword=keyword,
                resolved_landmark_id=picked_id,
            )
            nearby = await self._fetch_nearby(picked_id, query, max_pages)
            if nearby:
                log_event(
                    LOGGER,
                    "planner.degrade.landmark.retry_hit",
                    source_landmark=original_landmark,
                    resolved_landmark_id=picked_id,
                    item_count=len(nearby),
                )
                return nearby
        return []

    async def _degrade_to_by_platform(
        self,
        *,
        query: StructuredQuery,
        max_pages: int,
        from_route: str,
        reason: str,
    ) -> list[HouseLite]:
        by_platform = getattr(self.houses_client, "by_platform", None)
        if not callable(by_platform):
            return []
        log_event(
            LOGGER,
            "planner.degrade.by_platform",
            from_route=from_route,
            reason=reason,
        )
        return await self._fetch_by_platform(query, max_pages)


def _layout_to_bedrooms(layout: str | None) -> str | None:
    if not layout:
        return None
    normalized = layout.translate(str.maketrans({"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5"}))
    for ch in normalized:
        if ch.isdigit():
            return ch
    return None


def _to_elevator_param(value: bool | None) -> str | None:
    if value is None:
        return None
    return "true" if value else "false"


def _landmark_fuzzy_keywords(*, original_landmark: str, landmark_name: str | None) -> list[str]:
    items: list[str] = []
    if isinstance(landmark_name, str) and landmark_name.strip():
        items.append(landmark_name.strip())
    if isinstance(original_landmark, str) and original_landmark.strip():
        items.append(original_landmark.strip())

    seen: set[str] = set()
    keywords: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        keywords.append(item)
    return keywords


def _normalize_decoration_param(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None

    normalized_map = {
        "精装修": "精装",
        "精装": "精装",
        "简装修": "简装",
        "简装": "简装",
    }
    return normalized_map.get(cleaned)


def _has_landmark_id(item: Any) -> bool:
    value = getattr(item, "id", None)
    return isinstance(value, str) and bool(value.strip())


def _has_landmark_name(item: Any) -> bool:
    value = getattr(item, "name", None)
    return isinstance(value, str) and bool(value.strip())
