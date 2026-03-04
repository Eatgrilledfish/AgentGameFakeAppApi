from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.dialogue import DialogueManager
from app.agent.formatter import OutputFormatter
from app.agent.nlu import RuleBasedNLU
from app.agent.state import StateStore
from app.infra.cache import CacheManager
from app.schemas import CaseType, HouseLite, HouseViewModel, InvokeRequest, Landmark, Listing, SessionState, StructuredQuery
from app.settings import AgentSettings


@dataclass
class _Plan:
    plan_type: str
    landmark_id: str | None = None


class DummyPlanner:
    def __init__(self) -> None:
        self.executed_queries: list[StructuredQuery] = []

    async def build_plan(self, query: StructuredQuery) -> _Plan:
        if query.hard.landmark_name:
            return _Plan(plan_type="nearby_landmark", landmark_id="LM_1")
        return _Plan(plan_type="by_platform")

    async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
        self.executed_queries.append(query.model_copy(deep=True))
        if query.hard.landmark_name == "望京":
            return [
                HouseLite(
                    house_id="HF_WJ",
                    rent=7600,
                    layout="2居1厅1卫",
                    district="朝阳",
                    community="望京花园",
                    subway_distance=600,
                    commute_to_xierqi_min=45,
                    status="可租",
                )
            ]
        if query.hard.district == "大兴":
            return [
                HouseLite(
                    house_id="HF_DX",
                    rent=3900,
                    layout="2居1厅1卫",
                    district="大兴",
                    community="黄村小区",
                    subway_distance=900,
                    commute_to_xierqi_min=70,
                    status="可租",
                )
            ]
        if query.hard.district == "通州":
            return [
                HouseLite(
                    house_id="HF_TZ_1",
                    rent=3300,
                    layout="2居1厅1卫",
                    district="通州",
                    community="梨园",
                    subway_distance=1400,
                    commute_to_xierqi_min=80,
                    status="可租",
                )
            ]
        return [
            HouseLite(
                house_id="HF_4",
                rent=3800,
                layout="2居1厅1卫",
                district="大兴",
                community="天宫院",
                subway_distance=850,
                commute_to_xierqi_min=75,
                status="可租",
            )
        ]


class DummyRanker:
    async def rank_two_stage(self, candidates: list[HouseLite], query: StructuredQuery, max_output: int = 5) -> list[HouseViewModel]:
        rows: list[HouseViewModel] = []
        for house in candidates[:max_output]:
            rows.append(
                HouseViewModel(
                    house_id=house.house_id,
                    listing_platform="安居客",
                    rent=house.rent,
                    layout=house.layout,
                    district=house.district,
                    community=house.community,
                    subway_distance=house.subway_distance,
                    commute_to_xierqi_min=house.commute_to_xierqi_min,
                )
            )
        return rows


class DummyHousesClient:
    def __init__(self) -> None:
        self.listing_calls: list[str] = []
        self.detail_calls: list[str] = []
        self.rent_calls: list[tuple[str, str]] = []
        self.terminate_calls: list[tuple[str, str]] = []
        self.details = {
            "HF_4": HouseLite(
                house_id="HF_4",
                rent=3800,
                layout="2居1厅1卫",
                district="大兴",
                community="天宫院",
                subway_distance=850,
                commute_to_xierqi_min=75,
                status="可租",
            ),
            "HF_WJ": HouseLite(
                house_id="HF_WJ",
                rent=7600,
                layout="2居1厅1卫",
                district="朝阳",
                community="望京花园",
                subway_distance=600,
                commute_to_xierqi_min=45,
                status="可租",
            ),
            "HF_TZ_1": HouseLite(
                house_id="HF_TZ_1",
                rent=3300,
                layout="2居1厅1卫",
                district="通州",
                community="梨园",
                subway_distance=1400,
                commute_to_xierqi_min=80,
                status="可租",
            ),
        }

    async def init_houses(self) -> dict:
        return {"ok": True}

    async def get_house_detail(self, house_id: str) -> HouseLite | None:
        self.detail_calls.append(house_id)
        return self.details.get(house_id)

    async def get_listings(self, house_id: str) -> dict:
        self.listing_calls.append(house_id)
        return {
            "items": [
                Listing(listing_platform="安居客", rent=3800, status="可租"),
                Listing(listing_platform="链家", rent=3900, status="可租"),
                Listing(listing_platform="58同城", rent=4100, status="可租"),
            ]
        }

    async def rent(self, house_id: str, listing_platform: str) -> dict:
        self.rent_calls.append((house_id, listing_platform))
        return {"ok": True, "house_id": house_id, "listing_platform": listing_platform}

    async def terminate(self, house_id: str, listing_platform: str) -> dict:
        self.terminate_calls.append((house_id, listing_platform))
        return {"ok": True}

    async def offline(self, house_id: str, listing_platform: str) -> dict:
        return {"ok": True}


def _build_dialogue(
    planner: DummyPlanner | None = None,
    houses_client: DummyHousesClient | None = None,
    cache: CacheManager | None = None,
) -> tuple[DialogueManager, SessionState, DummyPlanner, DummyHousesClient]:
    settings = AgentSettings()
    state = SessionState(session_id="sess-ctx", user_id="u-1", case_type=CaseType.single)
    planner_impl = planner or DummyPlanner()
    houses_impl = houses_client or DummyHousesClient()
    cache_impl = cache or CacheManager(settings)
    dialogue = DialogueManager(
        state_store=StateStore(settings),
        nlu=RuleBasedNLU(),
        planner=planner_impl,
        ranker=DummyRanker(),
        formatter=OutputFormatter(),
        houses_client=houses_impl,
        cache=cache_impl,
        max_output_candidates=5,
    )
    return dialogue, state, planner_impl, houses_impl


@pytest.mark.asyncio
async def test_dialogue_handles_listing_then_rent_with_context() -> None:
    dialogue, state, _, houses = _build_dialogue()

    resp1 = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="帮我找大兴区两居室，预算4000以内的房子"),
        state,
        is_new_session=True,
    )
    assert resp1.candidates[0].house_id == "HF_DX"

    resp2 = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="HF_DX这套在安居客、链家、58同城上分别多少钱？"),
        state,
        is_new_session=False,
    )
    assert "安居客" in resp2.text
    assert houses.listing_calls[-1] == "HF_DX"

    resp3 = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="这套可以租，帮我办理租房"),
        state,
        is_new_session=False,
    )
    assert "已提交租房操作" in resp3.text
    assert houses.rent_calls[-1] == ("HF_DX", "安居客")


@pytest.mark.asyncio
async def test_dialogue_summarizes_top5_subway_distance_for_followup_question() -> None:
    class MultiPlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            return [
                HouseLite(house_id="HF_A", district="朝阳", subway_distance=320, rent=6500, layout="2居1厅1卫", status="可租"),
                HouseLite(house_id="HF_B", district="朝阳", subway_distance=780, rent=6200, layout="2居1厅1卫", status="可租"),
                HouseLite(house_id="HF_C", district="朝阳", subway_distance=1200, rent=6000, layout="2居1厅1卫", status="可租"),
            ]

    dialogue, state, _, _ = _build_dialogue(planner=MultiPlanner())

    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="帮我找朝阳区两居室，预算7000以内"),
        state,
        is_new_session=True,
    )
    resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="这套离地铁站多远？"),
        state,
        is_new_session=False,
    )

    assert resp.debug["response_kind"] == "detail"
    assert resp.debug["detail_mode"] == "top5_subway_distance"
    assert resp.debug["referenced_house_ids"] == ["HF_A", "HF_B", "HF_C"]
    assert "HF_A：离地铁约 320 米" in resp.text
    assert "HF_B：离地铁约 780 米" in resp.text
    assert "HF_C：离地铁约 1200 米" in resp.text


@pytest.mark.asyncio
async def test_dialogue_asks_follow_up_on_underconstrained_search() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses)

    resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="帮我找房"),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "clarify"
    assert "预算上限" in resp.text
    assert "区域、小区或地铁站" in resp.text
    assert planner.executed_queries == []


@pytest.mark.asyncio
async def test_dialogue_can_refer_to_first_search_snapshot() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, houses = _build_dialogue(planner=planner, houses_client=houses)

    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="我想在望京租一套两居室，预算8000以内，有电梯"),
        state,
        is_new_session=True,
    )
    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="换大兴区的看看吧，两居室，预算4000以内"),
        state,
        is_new_session=False,
    )
    resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="还是最开始望京那套比较好，它的详细情况怎么样？"),
        state,
        is_new_session=False,
    )

    assert houses.detail_calls[-1] == "HF_WJ"
    assert state.focus_house_id == "HF_WJ"
    assert "HF_WJ" in resp.text
    assert planner.executed_queries[1].hard.district == "大兴"
    assert planner.executed_queries[1].hard.landmark_name is None


@pytest.mark.asyncio
async def test_dialogue_supports_search_then_rent_first_then_terminate_last() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, houses = _build_dialogue(planner=planner, houses_client=houses)

    resp1 = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="通州两局商水商电房源有没有？"),
        state,
        is_new_session=True,
    )
    assert resp1.debug["response_kind"] == "search"
    assert resp1.candidates[0].house_id == "HF_TZ_1"
    assert planner.executed_queries[0].hard.utilities_type == "商水商电"

    resp2 = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="就租第一套吧。"),
        state,
        is_new_session=False,
    )
    assert "已提交租房操作" in resp2.text
    assert houses.rent_calls[-1] == ("HF_TZ_1", "安居客")

    resp3 = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="算了不租了，帮我退掉吧。"),
        state,
        is_new_session=False,
    )
    assert "已提交退租操作" in resp3.text
    assert houses.terminate_calls[-1] == ("HF_TZ_1", "安居客")


@pytest.mark.asyncio
async def test_dialogue_accepts_llm_intent_parse_override() -> None:
    dialogue, state, _, houses = _build_dialogue()

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="HF_4这套帮我租",
            meta={
                "llm_parse": {
                    "intent": "rent",
                    "params": {},
                    "tag_need": {"must": [], "avoid": [], "prefer": []},
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert "已提交租房操作" in resp.text
    assert houses.rent_calls[-1] == ("HF_4", "安居客")


@pytest.mark.asyncio
async def test_dialogue_prefers_llm_parse_even_with_low_confidence() -> None:
    dialogue, state, _, houses = _build_dialogue()

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="HF_4这套帮我租",
            meta={
                "llm_parse": {
                    "intent": "rent",
                    "params": {},
                    "tag_need": {"must": [], "avoid": [], "prefer": []},
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert "已提交租房操作" in resp.text
    assert houses.rent_calls[-1] == ("HF_4", "安居客")


@pytest.mark.asyncio
async def test_dialogue_prefers_llm_tool_plan_for_action() -> None:
    dialogue, state, _, houses = _build_dialogue()

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="HF_4这套帮我租",
            meta={
                "llm_parse": {
                    "intent": "rent",
                    "params": {},
                    "tag_need": {"must": [], "avoid": [], "prefer": []},
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert "已提交租房操作" in resp.text
    assert houses.rent_calls[-1] == ("HF_4", "安居客")


@pytest.mark.asyncio
async def test_dialogue_normalizes_llm_decoration_to_upstream_allowed_value() -> None:
    planner = DummyPlanner()
    dialogue, state, planner, _ = _build_dialogue(planner=planner)

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="帮我找大兴精装修的两居，预算4000以内",
            meta={
                "llm_parse": {
                    "intent": "search",
                    "params": {"district": "大兴", "bedrooms": "2", "max_price": 4000, "decoration": "精装修"},
                    "tag_need": {"must": [], "avoid": [], "prefer": []},
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    assert planner.executed_queries
    assert planner.executed_queries[-1].soft.decoration == "精装"


@pytest.mark.asyncio
async def test_dialogue_explicit_house_id_overrides_llm_context_reference() -> None:
    dialogue, state, _, houses = _build_dialogue()

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="HF_67这套可以租吗？",
            meta={
                "llm_parse": {
                    "intent": "rent",
                    "hard": {"house_id": "HF_199", "listing_platform": "安居客"},
                    "confidence": 0.92,
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert "未找到房源 HF_67" in resp.text
    assert houses.rent_calls == []


@pytest.mark.asyncio
async def test_dialogue_can_rent_question_checks_status_only() -> None:
    dialogue, state, _, houses = _build_dialogue()

    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="帮我找两居，预算4000以内"),
        state,
        is_new_session=True,
    )
    resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="这套不错，我可以租吗？"),
        state,
        is_new_session=False,
    )

    assert resp.debug["response_kind"] == "detail"
    assert "当前状态：可租" in resp.text
    assert houses.rent_calls == []


@pytest.mark.asyncio
async def test_dialogue_rent_requires_available_status() -> None:
    dialogue, state, _, houses = _build_dialogue()

    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="帮我找两居，预算4000以内"),
        state,
        is_new_session=True,
    )
    houses.details["HF_4"].status = "已租"
    resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="这套不错，我要租这套"),
        state,
        is_new_session=False,
    )

    assert resp.debug["response_kind"] == "detail"
    assert "暂不可直接办理租房" in resp.text
    assert houses.rent_calls == []


@pytest.mark.asyncio
async def test_dialogue_prefers_llm_tool_plan_for_search_arguments() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses)

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="帮我找房",
            meta={
                "llm_parse": {
                    "intent": "search",
                    "params": {"district": "大兴", "max_price": 4000, "bedrooms": "2"},
                    "tag_need": {"must": [], "avoid": [], "prefer": []},
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    assert resp.candidates[0].house_id == "HF_DX"
    assert planner.executed_queries[-1].hard.budget_max == 4000
    assert planner.executed_queries[-1].hard.layout in {"2居", "两居"}


@pytest.mark.asyncio
async def test_dialogue_maps_llm_business_area_in_district_field_to_area() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses)

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="我想在望京租一套两居室，预算8000以内，有电梯",
            meta={
                "llm_parse": {
                    "tool_plan": {
                        "operationId": "get_houses_by_platform",
                        "arguments": {
                            "district": "望京",
                            "max_price": 8000,
                            "bedrooms": "2",
                            "elevator": True,
                        },
                    },
                    "confidence": 0.9,
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    assert planner.executed_queries
    assert planner.executed_queries[-1].hard.district is None
    assert planner.executed_queries[-1].hard.area == "望京"


@pytest.mark.asyncio
async def test_dialogue_keeps_admin_division_in_district_from_llm_arguments() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses)

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="帮我找天河区两居，预算8000",
            meta={
                "llm_parse": {
                    "tool_plan": {
                        "operationId": "get_houses_by_platform",
                        "arguments": {
                            "district": "天河区",
                            "max_price": 8000,
                            "bedrooms": "2",
                        },
                    },
                    "confidence": 0.9,
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    assert planner.executed_queries
    assert planner.executed_queries[-1].hard.district == "天河"
    assert planner.executed_queries[-1].hard.area is None


@pytest.mark.asyncio
async def test_dialogue_uses_preloaded_landmark_catalog_to_disambiguate_district_field() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    cache = CacheManager(AgentSettings())
    cache.prime_landmark_catalog(
        [
            Landmark(id="LM_WJ", name="望京", category="landmark", district="朝阳"),
            Landmark(id="SS_001", name="车公庄站", category="subway", district="西城"),
        ],
        stats={"categories": ["company", "landmark", "subway"], "districts": ["朝阳", "西城", "大兴"]},
    )
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses, cache=cache)

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="按之前条件继续找房",
            meta={
                "llm_parse": {
                    "intent": "search",
                    "params": {"district": "望京", "max_price": 8000, "bedrooms": "2"},
                    "tag_need": {"must": [], "avoid": [], "prefer": []},
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    assert planner.executed_queries
    assert planner.executed_queries[-1].hard.district is None
    assert planner.executed_queries[-1].hard.area == "望京"


@pytest.mark.asyncio
async def test_dialogue_preserves_llm_nearby_max_distance_argument() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses)

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="车公庄站500米内的两居",
            meta={
                "llm_parse": {
                    "intent": "search",
                    "params": {"bedrooms": "2", "max_subway_dist": 500},
                    "tag_need": {"must": [], "avoid": [], "prefer": []},
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    assert planner.executed_queries[-1].hard.max_subway_dist == 500
    assert planner.executed_queries[-1].hard.layout in {"2居", "两居"}


@pytest.mark.asyncio
async def test_dialogue_complaint_stores_preferences_then_search_uses_them() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses)

    chat_resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="唉，我现在的房子住的不太舒服，采光不好，房间也小",
        ),
        state,
        is_new_session=True,
    )

    assert chat_resp.debug["response_kind"] == "chat"
    assert planner.executed_queries == []
    assert state.confirmed_constraints.area_min is None
    assert state.soft_preferences.orientation == "朝南"
    assert state.soft_preferences.prefer_spacious is True

    chat_resp2 = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="而且通勤时间也太长了，每天都要早起",
        ),
        state,
        is_new_session=False,
    )

    assert chat_resp2.debug["response_kind"] == "chat"
    assert state.confirmed_constraints.max_subway_dist is None
    assert state.soft_preferences.prioritize_subway_distance is True
    assert state.soft_preferences.prioritize_commute is True

    search_resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="想换个房子了，你能帮我找找吗？",
        ),
        state,
        is_new_session=False,
    )

    assert search_resp.debug["response_kind"] == "search"
    assert planner.executed_queries

    search_resp2 = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="预算6000，朝阳区",
        ),
        state,
        is_new_session=False,
    )

    assert search_resp2.debug["response_kind"] == "search"
    assert planner.executed_queries
    assert planner.executed_queries[-1].hard.area_min is None
    assert planner.executed_queries[-1].hard.max_subway_dist is None
    assert planner.executed_queries[-1].soft.orientation == "朝南"
    assert planner.executed_queries[-1].soft.prefer_spacious is True
    assert planner.executed_queries[-1].soft.prioritize_subway_distance is True
    assert planner.executed_queries[-1].soft.prioritize_commute is True


@pytest.mark.asyncio
async def test_dialogue_direct_requirements_start_search_and_keep_context_tags() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses)

    await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="现在住着不太舒服，采光不好，房间也小",
        ),
        state,
        is_new_session=True,
    )

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="帮我找朝阳区两居，预算6000，离地铁近一点",
        ),
        state,
        is_new_session=False,
    )

    assert resp.debug["response_kind"] == "search"
    assert planner.executed_queries
    assert planner.executed_queries[-1].hard.budget_max == 6000
    assert planner.executed_queries[-1].hard.layout in {"2居", "两居"}
    assert planner.executed_queries[-1].hard.district == "朝阳"
    assert planner.executed_queries[-1].soft.orientation == "朝南"


@pytest.mark.asyncio
async def test_dialogue_promotes_llm_chat_to_search_for_tag_refinement_with_active_context() -> None:
    class TagPlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            _ = plan
            _ = case_type
            return [
                HouseLite(
                    house_id="HF_ONLINE",
                    rent=3000,
                    layout="2居1厅1卫",
                    district="朝阳",
                    community="A",
                    subway_distance=500,
                    commute_to_xierqi_min=35,
                    status="可租",
                    tags=["仅线上VR看房", "近公园"],
                ),
                HouseLite(
                    house_id="HF_OFFLINE",
                    rent=2900,
                    layout="2居1厅1卫",
                    district="朝阳",
                    community="B",
                    subway_distance=480,
                    commute_to_xierqi_min=33,
                    status="可租",
                    tags=["仅线下看房", "近公园"],
                ),
            ]

    dialogue, state, planner, _ = _build_dialogue(planner=TagPlanner())

    first = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="帮我找朝阳区两居室，预算3000以内"),
        state,
        is_new_session=True,
    )
    assert first.debug["response_kind"] == "search"

    second = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="平时工作太忙，希望能线上VR看房，不用跑现场",
            meta={
                "llm_parse": {
                    "intent": "chat",
                    "tool_plan": {"operationId": "none", "arguments": {}},
                    "assistant_reply": "这里是一个聊天回复",
                    "confidence": 0.8,
                }
            },
        ),
        state,
        is_new_session=False,
    )

    assert second.debug["response_kind"] == "search"
    refined = planner.executed_queries[-1]
    assert any("线上" in tag for tag in refined.soft.preferred_tags)
    assert any("线下" in tag for tag in refined.soft.avoid_tags)


@pytest.mark.asyncio
async def test_dialogue_promotes_llm_house_detail_to_search_for_preference_refinement() -> None:
    class PreferencePlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            _ = plan
            _ = case_type
            return [
                HouseLite(
                    house_id="HF_MONTHLY",
                    rent=3200,
                    layout="2居1厅1卫",
                    district="朝阳",
                    community="A",
                    subway_distance=600,
                    commute_to_xierqi_min=36,
                    status="可租",
                    tags=["月付", "房东直租"],
                ),
                HouseLite(
                    house_id="HF_YEARLY",
                    rent=3000,
                    layout="2居1厅1卫",
                    district="朝阳",
                    community="B",
                    subway_distance=500,
                    commute_to_xierqi_min=34,
                    status="可租",
                    tags=["年付", "收中介费"],
                ),
            ]

    dialogue, state, planner, _ = _build_dialogue(planner=PreferencePlanner())

    first = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="帮我找朝阳区两居室，预算3000左右"),
        state,
        is_new_session=True,
    )
    assert first.debug["response_kind"] == "search"

    second = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="想问下能不能月付？最好能是房东直租。",
            meta={
                "llm_parse": {
                    "intent": "house_detail",
                    "params": {},
                    "tag_need": {"must": [], "avoid": ["年付", "收中介费"], "prefer": ["月付", "房东直租"]},
                }
            },
        ),
        state,
        is_new_session=False,
    )

    assert second.debug["response_kind"] == "search"
    refined = planner.executed_queries[-1]
    assert "月付" in refined.tag_need.prefer
    assert "房东直租" in refined.tag_need.prefer


@pytest.mark.asyncio
async def test_dialogue_context_continuation_can_start_search_with_previous_tags() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, planner, _ = _build_dialogue(planner=planner, houses_client=houses)

    await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="现在住着不太舒服，采光不好，房间也小，而且通勤时间长",
        ),
        state,
        is_new_session=True,
    )

    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-ctx",
            case_type=CaseType.single,
            message="那就按之前的条件继续找房吧",
        ),
        state,
        is_new_session=False,
    )

    assert resp.debug["response_kind"] == "search"
    assert planner.executed_queries
    assert planner.executed_queries[-1].hard.area_min is None
    assert planner.executed_queries[-1].hard.max_subway_dist is None
    assert planner.executed_queries[-1].soft.orientation == "朝南"
    assert planner.executed_queries[-1].soft.prefer_spacious is True
    assert planner.executed_queries[-1].soft.prioritize_subway_distance is True
    assert planner.executed_queries[-1].soft.prioritize_commute is True


@pytest.mark.asyncio
async def test_dialogue_treats_this_one_as_focus_not_first_rank() -> None:
    planner = DummyPlanner()
    houses = DummyHousesClient()
    dialogue, state, _, houses = _build_dialogue(planner=planner, houses_client=houses)

    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="我想在望京租一套两居室，预算8000以内，有电梯"),
        state,
        is_new_session=True,
    )
    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="换大兴区的看看吧，两居室，预算4000以内"),
        state,
        is_new_session=False,
    )
    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="还是最开始望京那套比较好，它的详细情况怎么样？"),
        state,
        is_new_session=False,
    )
    resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-ctx", case_type=CaseType.single, message="这套可以租吗？我想租这一套"),
        state,
        is_new_session=False,
    )

    assert "已提交租房操作" in resp.text
    assert houses.rent_calls[-1][0] == "HF_WJ"


@pytest.mark.asyncio
async def test_dialogue_multi_flow_search_compare_then_rent() -> None:
    class MultiPlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            _ = plan
            _ = case_type
            return [
                HouseLite(
                    house_id="HF_M1",
                    rent=6200,
                    layout="2居1厅1卫",
                    district="朝阳",
                    community="望京",
                    subway_distance=550,
                    commute_to_xierqi_min=42,
                    status="可租",
                ),
                HouseLite(
                    house_id="HF_M2",
                    rent=5800,
                    layout="2居1厅1卫",
                    district="朝阳",
                    community="酒仙桥",
                    subway_distance=900,
                    commute_to_xierqi_min=55,
                    status="可租",
                ),
                HouseLite(
                    house_id="HF_M3",
                    rent=6400,
                    layout="2居1厅1卫",
                    district="朝阳",
                    community="将台",
                    subway_distance=780,
                    commute_to_xierqi_min=50,
                    status="可租",
                ),
            ]

    planner = MultiPlanner()
    houses = DummyHousesClient()
    dialogue, state, _, houses = _build_dialogue(planner=planner, houses_client=houses)

    search_resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-multi", case_type=CaseType.multi, message="帮我找朝阳区两居，预算7000以内，通勤别太长"),
        state,
        is_new_session=True,
    )
    assert search_resp.debug["response_kind"] == "search"
    assert len(search_resp.candidates) == 3

    compare_resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-multi", case_type=CaseType.multi, message="把这几套对比一下，给我一个决策建议"),
        state,
        is_new_session=False,
    )
    assert compare_resp.debug["response_kind"] == "compare"
    assert compare_resp.candidates
    best_house_id = compare_resp.candidates[0].house_id
    assert state.focus_house_id == best_house_id

    rent_resp = await dialogue.handle_turn(
        InvokeRequest(session_id="sess-multi", case_type=CaseType.multi, message="这套不错，我要租这套"),
        state,
        is_new_session=False,
    )
    assert "已提交租房操作" in rent_resp.text
    assert houses.rent_calls[-1][0] == best_house_id


@pytest.mark.asyncio
async def test_dialogue_search_updates_session_tag_lexicon_memory() -> None:
    class TaggedPlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            _ = plan
            _ = case_type
            return [
                HouseLite(
                    house_id="HF_TAG_1",
                    rent=3800,
                    layout="2居1厅1卫",
                    district="大兴",
                    community="黄村小区",
                    subway_distance=780,
                    commute_to_xierqi_min=65,
                    status="可租",
                    tags=["免宽带费", "月付", "物业态度差"],
                )
            ]

    dialogue, state, _, _ = _build_dialogue(planner=TaggedPlanner(), houses_client=DummyHousesClient())
    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-tag-memory",
            case_type=CaseType.single,
            message="帮我找大兴区两居，预算4000以内，我希望宽带包含，不要物业态度差",
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    assert state.tag_lexicon
    assert "HF_TAG_1" in state.houses
    assert state.houses["HF_TAG_1"].tag_ids
    assert state.candidate_state.latest_house_ids == ["HF_TAG_1"]
    assert state.candidate_state.focus_house_id == "HF_TAG_1"
    assert any("宽带" in note or "物业" in note for note in state.req.soft.notes)


@pytest.mark.asyncio
async def test_dialogue_rejected_does_not_hard_drop_top2_when_conflict_not_certain() -> None:
    class ConflictPlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            _ = plan
            _ = case_type
            return [
                HouseLite(
                    house_id="HF_TOP1",
                    rent=3600,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=500,
                    commute_to_xierqi_min=40,
                    status="可租",
                    tags=["物业态度差", "月付"],
                ),
                HouseLite(
                    house_id="HF_TOP2",
                    rent=3650,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=520,
                    commute_to_xierqi_min=42,
                    status="可租",
                    tags=["月付"],
                ),
                HouseLite(
                    house_id="HF_TOP3",
                    rent=3800,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=600,
                    commute_to_xierqi_min=50,
                    status="可租",
                    tags=["月付"],
                ),
            ]

    dialogue, state, _, _ = _build_dialogue(planner=ConflictPlanner(), houses_client=DummyHousesClient())
    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-reject-protect",
            case_type=CaseType.single,
            message="帮我找大兴两居，预算4000以内，不要物业态度差，最好月付",
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    decisions = resp.debug["semantic_fusion"]["decisions"]
    assert decisions["HF_TOP1"]["action"] in {"rejected_penalty", "selected_boost", "must_confirm_boost"}
    assert "HF_TOP1" in [item.house_id for item in resp.candidates]


@pytest.mark.asyncio
async def test_dialogue_rejected_hard_conflict_can_drop_non_top2() -> None:
    class ConflictPlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            _ = plan
            _ = case_type
            return [
                HouseLite(house_id="HF_A", rent=3600, layout="2居1厅1卫", district="大兴", subway_distance=450, commute_to_xierqi_min=35, status="可租", tags=["月付"]),
                HouseLite(house_id="HF_B", rent=3650, layout="2居1厅1卫", district="大兴", subway_distance=500, commute_to_xierqi_min=40, status="可租", tags=["月付"]),
                HouseLite(house_id="HF_C", rent=3900, layout="2居1厅1卫", district="大兴", subway_distance=700, commute_to_xierqi_min=60, status="可租", tags=["物业态度差"]),
                HouseLite(house_id="HF_D", rent=3950, layout="2居1厅1卫", district="大兴", subway_distance=750, commute_to_xierqi_min=65, status="可租", tags=["月付"]),
            ]

    dialogue, state, _, _ = _build_dialogue(planner=ConflictPlanner(), houses_client=DummyHousesClient())
    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-reject-drop",
            case_type=CaseType.single,
            message="帮我找大兴两居，预算4000以内，不要物业态度差",
        ),
        state,
        is_new_session=True,
    )

    decisions = resp.debug["semantic_fusion"]["decisions"]
    assert decisions["HF_C"]["action"] == "rejected_drop"
    assert "HF_C" not in [item.house_id for item in resp.candidates]


@pytest.mark.asyncio
async def test_dialogue_candidate_filter_uses_tagfocus_relevant_subset_and_limit() -> None:
    many_tags = [f"宽带标签{i}" for i in range(1, 71)] + ["完全无关标签"]

    class ManyTagsPlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            _ = plan
            _ = case_type
            return [
                HouseLite(
                    house_id="HF_TAGS_1",
                    rent=3500,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=500,
                    commute_to_xierqi_min=40,
                    status="可租",
                    tags=many_tags,
                ),
                HouseLite(
                    house_id="HF_TAGS_2",
                    rent=3550,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=520,
                    commute_to_xierqi_min=42,
                    status="可租",
                    tags=[f"宽带标签{i}" for i in range(1, 11)],
                ),
                HouseLite(
                    house_id="HF_TAGS_3",
                    rent=3600,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=550,
                    commute_to_xierqi_min=45,
                    status="可租",
                    tags=["月付"],
                ),
            ]

    dialogue, state, _, _ = _build_dialogue(planner=ManyTagsPlanner(), houses_client=DummyHousesClient())
    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-relevant-limit",
            case_type=CaseType.single,
            message="帮我找大兴两居，预算4000以内，希望宽带包含",
        ),
        state,
        is_new_session=True,
    )

    relevant_ids = resp.debug["semantic_filter"]["relevant_tag_ids"]
    assert len(relevant_ids) <= 60
    irrelevant_tid = next((tid for tid, tag in state.tag_lexicon.items() if tag == "完全无关标签"), None)
    if irrelevant_tid:
        assert irrelevant_tid not in relevant_ids

    top_coverage_tids = [tid for tid, tag in state.tag_lexicon.items() if tag in {f"宽带标签{i}" for i in range(1, 11)}]
    assert top_coverage_tids
    assert all(tid in relevant_ids for tid in top_coverage_tids)


@pytest.mark.asyncio
async def test_dialogue_action_rejects_house_outside_latest_allowlist() -> None:
    dialogue, state, _, houses = _build_dialogue()
    await dialogue.handle_turn(
        InvokeRequest(session_id="sess-allowlist", case_type=CaseType.single, message="帮我找大兴区两居室，预算4000以内"),
        state,
        is_new_session=True,
    )
    resp = await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-allowlist",
            case_type=CaseType.single,
            message="帮我租HF_9999",
            meta={"llm_parse": {"intent": "rent", "hard": {"house_id": "HF_9999"}, "confidence": 0.9}},
        ),
        state,
        is_new_session=False,
    )

    assert "未在当前候选中找到 HF_9999" in resp.text
    assert houses.rent_calls == []


@pytest.mark.asyncio
async def test_dialogue_tag_need_accumulated_supports_revocation() -> None:
    class TaggedPlanner(DummyPlanner):
        async def execute_plan(self, plan: _Plan, query: StructuredQuery, case_type: CaseType) -> list[HouseLite]:
            self.executed_queries.append(query.model_copy(deep=True))
            _ = plan
            _ = case_type
            return [
                HouseLite(
                    house_id="HF_TAG_REVOKE",
                    rent=3600,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=700,
                    commute_to_xierqi_min=55,
                    status="可租",
                    tags=["物业态度差", "月付"],
                ),
                HouseLite(
                    house_id="HF_TAG_REVOKE_2",
                    rent=3650,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=720,
                    commute_to_xierqi_min=58,
                    status="可租",
                    tags=["月付"],
                ),
                HouseLite(
                    house_id="HF_TAG_REVOKE_3",
                    rent=3700,
                    layout="2居1厅1卫",
                    district="大兴",
                    subway_distance=730,
                    commute_to_xierqi_min=60,
                    status="可租",
                    tags=["月付"],
                ),
            ]

    dialogue, state, _, _ = _build_dialogue(planner=TaggedPlanner(), houses_client=DummyHousesClient())
    await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-tag-revoke",
            case_type=CaseType.single,
            message="帮我找大兴两居，预算4000以内，不要物业态度差",
        ),
        state,
        is_new_session=True,
    )
    assert any("物业" in item for item in state.req.soft.tag_need_accumulated.avoid)

    await dialogue.handle_turn(
        InvokeRequest(
            session_id="sess-tag-revoke",
            case_type=CaseType.single,
            message="继续找，物业无所谓",
        ),
        state,
        is_new_session=False,
    )
    assert not any("物业" in item for item in state.req.soft.tag_need_accumulated.avoid)
