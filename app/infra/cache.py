from __future__ import annotations

from cachetools import TTLCache

from app.settings import AgentSettings


class CacheManager:
    def __init__(self, settings: AgentSettings) -> None:
        self.landmark_by_name = TTLCache(maxsize=10_000, ttl=settings.cache.landmark_by_name_ttl_sec)
        self.community_amenities = TTLCache(maxsize=50_000, ttl=settings.cache.community_amenities_ttl_sec)
        self.house_detail = TTLCache(maxsize=50_000, ttl=settings.cache.house_detail_ttl_sec)
        self.house_listings = TTLCache(maxsize=50_000, ttl=settings.cache.house_listings_ttl_sec)
        self.query_result_ids = TTLCache(maxsize=10_000, ttl=settings.cache.query_result_ids_ttl_sec)

    def invalidate_house(self, house_id: str) -> None:
        self.house_detail.pop(house_id, None)
        self.house_listings.pop(house_id, None)

    def invalidate_query_cache(self) -> None:
        self.query_result_ids.clear()

    def invalidate_all_houses(self) -> None:
        self.house_detail.clear()
        self.house_listings.clear()
        self.query_result_ids.clear()
