from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.dialogue import DialogueManager
from app.agent.formatter import OutputFormatter
from app.agent.nlu import RuleBasedNLU
from app.agent.state import StateStore
from app.infra.cache import CacheManager
from app.schemas import CaseType, HouseLite, HouseViewModel, InvokeRequest, Listing, SessionState, StructuredQuery
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
) -> tuple[DialogueManager, SessionState, DummyPlanner, DummyHousesClient]:
    settings = AgentSettings()
    state = SessionState(session_id="sess-ctx", user_id="u-1", case_type=CaseType.single)
    planner_impl = planner or DummyPlanner()
    houses_impl = houses_client or DummyHousesClient()
    dialogue = DialogueManager(
        state_store=StateStore(settings),
        nlu=RuleBasedNLU(),
        planner=planner_impl,
        ranker=DummyRanker(),
        formatter=OutputFormatter(),
        houses_client=houses_impl,
        cache=CacheManager(settings),
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
            message="帮我处理一下这套",
            meta={
                "llm_parse": {
                    "intent": "rent",
                    "hard": {"house_id": "HF_4", "listing_platform": "安居客"},
                    "confidence": 0.93,
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
            message="帮我处理这个",
            meta={
                "llm_parse": {
                    "intent": "rent",
                    "hard": {"house_id": "HF_4", "listing_platform": "安居客"},
                    "confidence": 0.1,
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
            message="帮我处理这个",
            meta={
                "llm_parse": {
                    "tool_plan": {
                        "operationId": "rent_house",
                        "arguments": {"house_id": "HF_4", "listing_platform": "安居客"},
                    },
                    "confidence": 0.91,
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert "已提交租房操作" in resp.text
    assert houses.rent_calls[-1] == ("HF_4", "安居客")


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

    assert "已提交租房操作" in resp.text
    assert houses.rent_calls[-1] == ("HF_67", "安居客")


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
                    "tool_plan": {
                        "operationId": "get_houses_by_platform",
                        "arguments": {
                            "district": "大兴",
                            "max_price": 4000,
                            "bedrooms": "2",
                            "utilities_type": "商水商电",
                        },
                    },
                    "confidence": 0.89,
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert resp.debug["response_kind"] == "search"
    assert resp.candidates[0].house_id == "HF_DX"
    assert planner.executed_queries[-1].hard.budget_max == 4000
    assert planner.executed_queries[-1].hard.layout == "2居"
    assert planner.executed_queries[-1].hard.utilities_type == "商水商电"


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
                    "tool_plan": {
                        "operationId": "get_houses_nearby",
                        "arguments": {
                            "landmark_id": "车公庄站",
                            "max_distance": 500,
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
    assert planner.executed_queries[-1].hard.landmark_id == "车公庄站"
    assert planner.executed_queries[-1].hard.max_distance == 500
    assert planner.executed_queries[-1].hard.layout == "2居"


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
            meta={
                "llm_parse": {
                    "tool_plan": {
                        "operationId": "get_houses_by_platform",
                        "arguments": {
                            "orientation": "朝南",
                            "min_area": 60,
                        },
                    },
                    "confidence": 0.9,
                }
            },
        ),
        state,
        is_new_session=True,
    )

    assert chat_resp.debug["response_kind"] == "chat"
    assert planner.executed_queries == []
    assert state.confirmed_constraints.area_min == 60
    assert state.soft_preferences.orientation == "朝南"

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
    assert state.confirmed_constraints.max_subway_dist == 800

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
    assert planner.executed_queries[-1].hard.area_min == 60
    assert planner.executed_queries[-1].hard.max_subway_dist == 800
    assert planner.executed_queries[-1].soft.orientation == "朝南"


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
