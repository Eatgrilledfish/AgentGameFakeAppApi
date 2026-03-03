from __future__ import annotations

import time
from typing import Any

from cachetools import TTLCache

from app.schemas import Landmark
from app.settings import AgentSettings


class CacheManager:
    def __init__(self, settings: AgentSettings) -> None:
        self.landmark_by_name = TTLCache(maxsize=10_000, ttl=settings.cache.landmark_by_name_ttl_sec)
        self.community_amenities = TTLCache(maxsize=50_000, ttl=settings.cache.community_amenities_ttl_sec)
        self.house_detail = TTLCache(maxsize=50_000, ttl=settings.cache.house_detail_ttl_sec)
        self.house_listings = TTLCache(maxsize=50_000, ttl=settings.cache.house_listings_ttl_sec)
        self.query_result_ids = TTLCache(maxsize=10_000, ttl=settings.cache.query_result_ids_ttl_sec)

        # Preloaded landmark catalog for fast disambiguation (district vs landmark/business area).
        self.landmark_name_aliases: set[str] = set()
        self.landmark_district_aliases: set[str] = set()
        self.landmark_category_by_name: dict[str, str] = {}
        self.landmark_categories: set[str] = set()
        self.landmark_catalog_loaded_at: float | None = None

    def prime_landmark_catalog(self, landmarks: list[Landmark], stats: dict[str, Any] | None = None) -> None:
        names: set[str] = set()
        district_aliases: set[str] = set()
        category_by_name: dict[str, str] = {}
        categories: set[str] = set()

        for item in landmarks:
            if not isinstance(item, Landmark):
                continue

            name = (item.name or "").strip()
            category = (item.category or "").strip()
            district = (item.district or "").strip()

            if district:
                district_aliases.add(district)
                if district.endswith("区") and len(district) > 1:
                    district_aliases.add(district[:-1])
            if category:
                categories.add(category)
            if not name:
                continue

            names.add(name)
            if category:
                category_by_name[name] = category
            if name.endswith("站") and len(name) > 1:
                name_no_station = name[:-1]
                names.add(name_no_station)
                if category:
                    category_by_name.setdefault(name_no_station, category)

            # Pre-fill planner cache.
            self.landmark_by_name[name] = item

        self._merge_stats_into_catalog(stats=stats, district_aliases=district_aliases, categories=categories)
        self.landmark_name_aliases = names
        self.landmark_district_aliases = district_aliases
        self.landmark_category_by_name = category_by_name
        self.landmark_categories = categories
        self.landmark_catalog_loaded_at = time.time()

    @staticmethod
    def _merge_stats_into_catalog(
        *,
        stats: dict[str, Any] | None,
        district_aliases: set[str],
        categories: set[str],
    ) -> None:
        if not isinstance(stats, dict):
            return

        possible_district_sources = [
            stats.get("districts"),
            stats.get("district_list"),
            stats.get("district_names"),
        ]
        district_counts = stats.get("district_counts")
        if isinstance(district_counts, dict):
            possible_district_sources.append(list(district_counts.keys()))
        for source in possible_district_sources:
            if not isinstance(source, list):
                continue
            for item in source:
                text = str(item).strip()
                if not text:
                    continue
                district_aliases.add(text)
                if text.endswith("区") and len(text) > 1:
                    district_aliases.add(text[:-1])

        possible_category_sources = [
            stats.get("categories"),
            stats.get("category_list"),
            stats.get("category_names"),
        ]
        category_counts = stats.get("category_counts")
        if isinstance(category_counts, dict):
            possible_category_sources.append(list(category_counts.keys()))
        for source in possible_category_sources:
            if not isinstance(source, list):
                continue
            for item in source:
                text = str(item).strip()
                if text:
                    categories.add(text)

    def invalidate_house(self, house_id: str) -> None:
        self.house_detail.pop(house_id, None)
        self.house_listings.pop(house_id, None)

    def invalidate_query_cache(self) -> None:
        self.query_result_ids.clear()

    def invalidate_all_houses(self) -> None:
        self.house_detail.clear()
        self.house_listings.clear()
        self.query_result_ids.clear()
