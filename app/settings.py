from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class TimeoutConfig:
    connect: float = 0.3
    read: float = 0.8
    write: float = 0.8
    pool: float = 0.2


@dataclass(slots=True)
class LimitsConfig:
    max_connections: int = 20
    max_keepalive_connections: int = 10


@dataclass(slots=True)
class CacheConfig:
    landmark_by_name_ttl_sec: int = 24 * 3600
    community_amenities_ttl_sec: int = 600
    house_detail_ttl_sec: int = 300
    house_listings_ttl_sec: int = 300
    query_result_ids_ttl_sec: int = 120


@dataclass(slots=True)
class RankingWeights:
    commute: float = 25.0
    rent: float = 25.0
    subway: float = 15.0
    layout_area: float = 15.0
    quality: float = 10.0
    amenities: float = 5.0
    listing_consistency: float = 5.0


@dataclass(slots=True)
class AgentSettings:
    api_base_url: str = "http://127.0.0.1:8080"
    default_user_id: str = "demo-user"
    app_host: str = "0.0.0.0"
    app_port: int = 8191
    request_deadline_ms: int = 4500
    enrich_listing_top_n: int = 20
    enrich_amenities_top_n: int = 5
    enrich_concurrency: int = 8
    max_pages_single: int = 2
    max_pages_multi: int = 3
    max_output_candidates: int = 5
    state_ttl_sec: int = 1800
    budget_limit_slices: int = 300
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    weights: RankingWeights = field(default_factory=RankingWeights)


def load_settings() -> AgentSettings:
    return AgentSettings(
        api_base_url=os.getenv("API_BASE_URL", "http://127.0.0.1:8080"),
        default_user_id=os.getenv("DEFAULT_USER_ID", "demo-user"),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8191")),
        request_deadline_ms=int(os.getenv("REQUEST_DEADLINE_MS", "4500")),
        enrich_listing_top_n=int(os.getenv("ENRICH_LISTING_TOP_N", "20")),
        enrich_amenities_top_n=int(os.getenv("ENRICH_AMENITIES_TOP_N", "5")),
        enrich_concurrency=int(os.getenv("ENRICH_CONCURRENCY", "8")),
        max_pages_single=int(os.getenv("MAX_PAGES_SINGLE", "2")),
        max_pages_multi=int(os.getenv("MAX_PAGES_MULTI", "3")),
        max_output_candidates=int(os.getenv("MAX_OUTPUT_CANDIDATES", "5")),
        state_ttl_sec=int(os.getenv("STATE_TTL_SEC", "1800")),
        budget_limit_slices=int(os.getenv("BUDGET_LIMIT_SLICES", "300")),
    )
