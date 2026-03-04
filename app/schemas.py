from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class CaseType(str, Enum):
    chat = "Chat"
    single = "Single"
    multi = "Multi"


class IntentType(str, Enum):
    chat = "chat"
    search = "search"
    compare = "compare"
    amenities = "amenities"
    house_detail = "house_detail"
    listings = "listings"
    rent_check = "rent_check"
    rent = "rent"
    terminate = "terminate"
    offline = "offline"


class Platform(str, Enum):
    lianjia = "链家"
    anjuke = "安居客"
    wuba = "58同城"


class HardConstraints(BaseModel):
    district: str | None = None
    area: str | None = None
    community: str | None = None
    landmark_id: str | None = None
    landmark_name: str | None = None
    landmark_category: str | None = None
    budget_min: int | None = None
    budget_max: int | None = None
    rent_type: str | None = None
    layout: str | None = None
    area_min: float | None = None
    max_subway_dist: int | None = None
    max_distance: int | None = None
    max_commute_min: int | None = None
    utilities_type: str | None = None
    move_in_date: str | None = None
    listing_platform: Platform | None = None
    house_id: str | None = None


class SoftPreferences(BaseModel):
    decoration: str | None = None
    elevator: bool | None = None
    orientation: str | None = None
    noise_preference: str | None = None
    amenities: list[str] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)
    avoid_tags: list[str] = Field(default_factory=list)
    value_for_money: bool | None = None
    prefer_spacious: bool = False
    prioritize_subway_distance: bool = False
    prioritize_commute: bool = False


class TagNeed(BaseModel):
    must: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    prefer: list[str] = Field(default_factory=list)


class StructuredQuery(BaseModel):
    intent: IntentType = IntentType.search
    hard: HardConstraints = Field(default_factory=HardConstraints)
    soft: SoftPreferences = Field(default_factory=SoftPreferences)
    tag_need: TagNeed = Field(default_factory=TagNeed)
    clarify_questions: list[str] = Field(default_factory=list)
    confidence: float = 0.6


class BudgetState(BaseModel):
    limit_slices: int = 300
    used_slices: int = 0
    used_tokens: int = 0
    llm_calls: int = 0


class SessionPhase(str, Enum):
    chatting = "CHATTING"
    slot_filling = "SLOT_FILLING"
    searching = "SEARCHING"
    presenting = "PRESENTING"
    refining = "REFINING"
    executing = "EXECUTING"


class HouseLite(BaseModel):
    house_id: str
    rent: int | None = None
    layout: str | None = None
    area: float | None = None
    business_area: str | None = None
    district: str | None = None
    community: str | None = None
    subway_distance: int | None = None
    commute_to_xierqi_min: int | None = None
    status: str | None = None
    tags: list[str] = Field(default_factory=list)
    decoration: str | None = None
    elevator: bool | None = None
    orientation: str | None = None
    available_date: str | None = None
    listing_platform: str | None = None
    distance_to_landmark: float | None = None
    walking_distance: float | None = None
    walking_duration: int | None = None
    hidden_noise_level: str | None = None


class Listing(BaseModel):
    listing_platform: str | None = None
    rent: int | None = None
    status: str | None = None
    url: str | None = None


class NearbyLandmark(BaseModel):
    name: str | None = None
    category: str | None = None
    distance_m: float | None = None


class Landmark(BaseModel):
    id: str | None = None
    name: str | None = None
    category: str | None = None
    district: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class HouseViewModel(BaseModel):
    house_id: str
    listing_platform: str | None = None
    rent: int | None = None
    layout: str | None = None
    area: float | None = None
    district: str | None = None
    community: str | None = None
    nearest_subway: str | None = None
    subway_distance: int | None = None
    commute_to_xierqi_min: int | None = None
    available_date: str | None = None
    tags: list[str] = Field(default_factory=list)
    pet_friendly: bool | None = None
    amenity_summary: dict[str, Any] = Field(default_factory=dict)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    score: float | None = None


class SearchSnapshot(BaseModel):
    query_text: str
    district: str | None = None
    area: str | None = None
    community: str | None = None
    landmark_name: str | None = None
    house_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TurnSummary(BaseModel):
    user: str
    assistant: str
    intent: IntentType | None = None
    house_ids: list[str] = Field(default_factory=list)


class SessionReqSoft(BaseModel):
    notes: list[str] = Field(default_factory=list)
    tag_need_accumulated: TagNeed = Field(default_factory=TagNeed)


class SessionReq(BaseModel):
    hard: HardConstraints = Field(default_factory=HardConstraints)
    soft: SessionReqSoft = Field(default_factory=SessionReqSoft)


class SessionHouseMemory(BaseModel):
    tag_ids: list[str] = Field(default_factory=list)
    price: int | None = None
    subway_distance: int | None = None
    rental_type: str | None = None
    area_sqm: float | None = None
    updated_ts: int | None = None


class CandidateState(BaseModel):
    latest_house_ids: list[str] = Field(default_factory=list)
    focus_house_id: str | None = None


class SessionState(BaseModel):
    session_id: str
    user_id: str
    case_type: CaseType
    phase: SessionPhase = SessionPhase.chatting
    confirmed_constraints: HardConstraints = Field(default_factory=HardConstraints)
    soft_preferences: SoftPreferences = Field(default_factory=SoftPreferences)
    unresolved_slots: list[str] = Field(default_factory=list)
    last_query_hash: str | None = None
    last_candidates: list[HouseLite] = Field(default_factory=list)
    house_context_top10: list[HouseLite] = Field(default_factory=list)
    last_top5: list[HouseViewModel] = Field(default_factory=list)
    search_history: list[SearchSnapshot] = Field(default_factory=list)
    focus_house_id: str | None = None
    focus_listing_platform: Platform | None = None
    recent_turns: list[TurnSummary] = Field(default_factory=list)
    conversation_summary: str = ""
    excluded_reasons: dict[str, str] = Field(default_factory=dict)
    req: SessionReq = Field(default_factory=SessionReq)
    tag_lexicon: dict[str, str] = Field(default_factory=dict)
    reverse_lexicon: dict[str, str] = Field(default_factory=dict)
    houses: dict[str, SessionHouseMemory] = Field(default_factory=dict)
    candidate_state: CandidateState = Field(default_factory=CandidateState)
    budget: BudgetState = Field(default_factory=BudgetState)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InvokeRequest(BaseModel):
    session_id: str
    case_type: CaseType = CaseType.single
    user_id: str | None = None
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class InvokeResponse(BaseModel):
    text: str
    candidates: list[HouseViewModel] = Field(default_factory=list)
    clarify_questions: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    model_ip: str
    session_id: str
    message: str = Field(validation_alias=AliasChoices("message", "content"))


class ChatResponse(BaseModel):
    session_id: str
    response: str
    status: str = "success"
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: int
    duration_ms: int


class HealthResponse(BaseModel):
    ok: bool = True
    service: str = "smart-rental-agent"
    now: date = Field(default_factory=date.today)
