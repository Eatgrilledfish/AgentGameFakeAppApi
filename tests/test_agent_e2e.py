import logging
import asyncio
import json
from fastapi.testclient import TestClient

import httpx

from app.clients.houses import HousesClient
from app.infra.cache import CacheManager
from app.infra.tool_recorder import record_tool_result
from app.main import _build_llm_context_facts, _load_nlu_tag_catalog, _preload_landmark_catalog, create_app
from app.schemas import CaseType, HouseLite, HouseViewModel, IntentType, InvokeResponse, Landmark, SearchSnapshot, SessionState, TurnSummary
from app.settings import AgentSettings


def _response_text(resp) -> str:
    return resp.json()["response"]


def _response_json(resp) -> dict:
    return json.loads(_response_text(resp))


def test_invoke_route_with_stub_service() -> None:
    app = create_app()

    class StubService:
        async def handle(self, request):
            return InvokeResponse(text="ok", candidates=[])

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/invoke",
            json={
                "session_id": "sess-1",
                "case_type": "Single",
                "user_id": "u-1",
                "message": "你好",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["text"] == "ok"


def test_chat_route_returns_houses_json_when_candidates_exist() -> None:
    app = create_app()

    class StubService:
        async def handle(self, request):
            return InvokeResponse(
                text="给你筛选好了",
                candidates=[
                    {
                        "house_id": "HF_2101",
                        "rent": 5000,
                    }
                ],
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-1", "message": "帮我找房"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess-1"
        response_payload = json.loads(body["response"])
        assert response_payload["message"] == "给你筛选好了"
        assert response_payload["houses"] == ["HF_2101"]
        assert body["status"] == "success"
    assert body["timestamp"] < 10_000_000_000


def test_llm_context_facts_include_soft_preferences() -> None:
    class StubState:
        focus_house_id = None
        focus_listing_platform = None
        confirmed_constraints = type(
            "Hard",
            (),
            {"model_dump": staticmethod(lambda exclude_none=True: {"max_subway_dist": 800, "area_min": 60})},
        )()
        soft_preferences = type(
            "Soft",
            (),
            {"model_dump": staticmethod(lambda exclude_none=True: {"orientation": "朝南", "prioritize_subway_distance": True})},
        )()
        search_history = []
        recent_turns = []

    facts = _build_llm_context_facts(StubState())

    assert facts["confirmed_constraints"]["max_subway_dist"] == 800
    assert facts["soft_preferences"]["orientation"] == "朝南"
    assert facts["soft_preferences"]["prioritize_subway_distance"] is True


def test_llm_context_facts_include_compact_top_houses_with_amenities() -> None:
    class StubHouse:
        house_id = "HF_2515"
        district = "朝阳"
        community = "融创苑8区"
        layout = "2居1厅1卫"
        rent = 3000
        subway_distance = 3635
        commute_to_xierqi_min = 39
        pet_friendly = True
        amenity_summary = {"shopping_count": 3, "nearest_shopping_m": 450, "park_count": 2, "nearest_park_m": 380}
        tags = ["可养狗", "近公园", "近商超", "网费另付"]

    class StubState:
        case_type = type("CaseType", (), {"value": "Multi"})()
        focus_house_id = "HF_2515"
        focus_listing_platform = None
        confirmed_constraints = type("Hard", (), {"model_dump": staticmethod(lambda exclude_none=True: {})})()
        soft_preferences = type(
            "Soft",
            (),
            {"model_dump": staticmethod(lambda exclude_none=True: {"preferred_tags": ["可养狗"], "amenities": ["公园"]})},
        )()
        search_history = []
        recent_turns = []
        last_top5 = [StubHouse()]

    facts = _build_llm_context_facts(StubState())

    assert "latest_top_houses" in facts
    assert facts["latest_top_houses"][0]["house_id"] == "HF_2515"
    assert facts["latest_top_houses"][0]["pet_friendly"] is True
    assert facts["latest_top_houses"][0]["amenity_summary"]["park_count"] == 2
    assert "近公园" in facts["latest_top_houses"][0]["key_tags"]


def test_startup_preload_landmark_catalog_primes_alias_dictionary() -> None:
    cache = CacheManager(AgentSettings())

    class StubLandmarksClient:
        async def list_landmarks(self):
            return [
                Landmark(id="LM_WJ", name="望京", category="landmark", district="朝阳"),
                Landmark(id="SS_001", name="车公庄站", category="subway", district="西城"),
            ]

        async def stats(self):
            return {
                "categories": ["company", "landmark", "subway"],
                "districts": ["朝阳", "西城", "海淀"],
            }

    asyncio.run(_preload_landmark_catalog(cache=cache, landmarks_client=StubLandmarksClient()))
    assert "望京" in cache.landmark_name_aliases
    assert "车公庄站" in cache.landmark_name_aliases
    assert "车公庄" in cache.landmark_name_aliases
    assert "朝阳" in cache.landmark_district_aliases
    assert "subway" in cache.landmark_categories


def test_startup_preload_landmark_calls_are_logged_into_agent_io(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "agent_http_io.log"
    monkeypatch.setenv("AGENT_HTTP_IO_LOG_PATH", str(log_file))

    async def _stub_http_get(self, url, headers=None):
        _ = self
        _ = headers
        request = httpx.Request("GET", url)
        if url.endswith("/api/landmarks"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "items": [
                            {"id": "LM_WJ", "name": "望京", "category": "landmark", "district": "朝阳"},
                        ]
                    }
                },
                request=request,
            )
        if url.endswith("/api/landmarks/stats"):
            return httpx.Response(
                200,
                json={"data": {"categories": ["landmark"], "districts": ["朝阳"]}},
                request=request,
            )
        return httpx.Response(404, json={"error": "not found"}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "get", _stub_http_get)

    app = create_app()
    with TestClient(app):
        pass

    content = log_file.read_text(encoding="utf-8")
    assert '"event": "http.agent_io.api.request"' in content
    assert '"url": "http://127.0.0.1:8080/api/landmarks"' in content
    assert '"session_id": "startup_landmarks_preload"' in content


def test_chat_route_accepts_content_field_as_user_input() -> None:
    app = create_app()
    captured: dict = {}

    class StubService:
        async def handle(self, request):
            captured["message"] = request.message
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "action"})

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-content-field", "content": "这是content字段"},
        )

    assert resp.status_code == 200
    assert captured["message"] == "这是content字段"
    assert _response_text(resp) == "ok"


def test_chat_route_uses_single_case_type_for_new_session() -> None:
    app = create_app()
    captured: dict = {}

    class StubService:
        async def handle(self, request):
            captured["case_type"] = request.case_type.value
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "chat"})

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-case-type-new", "message": "你好"},
        )

    assert resp.status_code == 200
    assert captured["case_type"] == "Single"


def test_chat_route_uses_multi_case_type_for_existing_session() -> None:
    app = create_app()
    captured: dict = {}

    class StubState:
        conversation_summary = "之前聊过预算和区域"

    class StubStateStore:
        def get(self, session_id):
            assert session_id == "sess-case-type-existing"
            return StubState()

    class StubService:
        state_store = StubStateStore()

        async def handle(self, request):
            captured["case_type"] = request.case_type.value
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "chat"})

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-case-type-existing", "message": "你好"},
        )

    assert resp.status_code == 200
    assert captured["case_type"] == "Multi"


def test_chat_route_returns_response_object_for_search_without_candidates() -> None:
    app = create_app()

    class StubService:
        async def handle(self, request):
            return InvokeResponse(
                text="当前条件无结果",
                candidates=[],
                debug={"response_kind": "search"},
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-2", "message": "帮我找房"},
        )

        assert resp.status_code == 200
        payload = _response_json(resp)
        assert payload == {"message": "当前条件无结果", "houses": []}


def test_chat_route_house_detail_response_must_include_houses_array() -> None:
    app = create_app()

    class StubService:
        async def handle(self, request):
            return InvokeResponse(
                text="HF_4 当前可租，离地铁约 620 米。",
                candidates=[{"house_id": "HF_4"}],
                debug={"response_kind": "detail", "intent": "house_detail", "referenced_house_ids": ["HF_4"]},
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-detail-houses", "message": "这套能租吗"},
        )

    assert resp.status_code == 200
    payload = _response_json(resp)
    assert payload["message"].startswith("HF_4 当前可租")
    assert payload["houses"] == ["HF_4"]


def test_chat_route_simple_greeting_skips_llm_nlu_call() -> None:
    app = create_app()
    captured = {"llm_calls": 0}

    class StubService:
        async def handle(self, request):
            return InvokeResponse(text="你好，我在。", candidates=[], debug={"response_kind": "chat", "intent": "chat"})

    class StubHttpClient:
        async def post(self, url, json, headers):
            _ = url
            _ = json
            _ = headers
            captured["llm_calls"] += 1
            raise AssertionError("simple greeting should not call llm nlu")

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-greeting-skip-llm", "message": "你好"},
        )

    assert resp.status_code == 200
    assert captured["llm_calls"] == 0
    assert _response_text(resp) == "你好，我在。"


def test_chat_route_rent_related_intents_return_houses_array() -> None:
    app = create_app()
    calls = {"idx": 0}

    class StubService:
        async def handle(self, request):
            calls["idx"] += 1
            if calls["idx"] == 1:
                return InvokeResponse(
                    text="HF_4 当前状态：可租。",
                    candidates=[{"house_id": "HF_4"}],
                    debug={"response_kind": "detail", "intent": "rent_check", "referenced_house_ids": ["HF_4"]},
                )
            return InvokeResponse(
                text="已提交租房操作：HF_4（安居客）。",
                candidates=[{"house_id": "HF_4"}],
                debug={"response_kind": "action", "intent": "rent", "referenced_house_ids": ["HF_4"]},
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp_check = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-rent-check", "message": "这套能租吗"},
        )
        resp_rent = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-rent-check", "message": "我要租这套"},
        )

    assert resp_check.status_code == 200
    assert _response_json(resp_check)["houses"] == ["HF_4"]
    assert resp_rent.status_code == 200
    assert _response_json(resp_rent)["houses"] == ["HF_4"]


def test_chat_route_chat_intent_in_housing_context_still_returns_houses() -> None:
    app = create_app()

    class StubState:
        conversation_summary = "上轮推荐过房源"
        candidate_state = type("CandidateState", (), {"latest_house_ids": ["HF_CTX_1", "HF_CTX_2"], "focus_house_id": None})()
        last_top5 = []
        houses = {}

    class StubStateStore:
        def get(self, session_id):
            assert session_id == "sess-ctx-chat"
            return StubState()

    class StubService:
        state_store = StubStateStore()

        async def handle(self, request):
            return InvokeResponse(text="支持VR看房。", candidates=[], debug={"response_kind": "chat", "intent": "chat"})

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-ctx-chat", "message": "这套能VR看房吗"},
        )

    assert resp.status_code == 200
    assert _response_json(resp)["houses"] == ["HF_CTX_1", "HF_CTX_2"]


def test_chat_route_generic_chat_word_does_not_force_houses_in_context() -> None:
    app = create_app()

    class StubState:
        conversation_summary = "上轮推荐过房源"
        candidate_state = type("CandidateState", (), {"latest_house_ids": ["HF_CTX_1", "HF_CTX_2"], "focus_house_id": None})()
        last_top5 = []
        houses = {}

    class StubStateStore:
        def get(self, session_id):
            assert session_id == "sess-ctx-chat-generic"
            return StubState()

    class StubService:
        state_store = StubStateStore()

        async def handle(self, request):
            return InvokeResponse(text="我挺好的。", candidates=[], debug={"response_kind": "chat", "intent": "chat"})

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-ctx-chat-generic", "message": "最近怎么样"},
        )

    assert resp.status_code == 200
    assert _response_text(resp) == "我挺好的。"


def test_chat_route_keeps_natural_text_for_non_search_reply() -> None:
    app = create_app()

    class StubService:
        async def handle(self, request):
            return InvokeResponse(
                text="HF_4 各平台挂牌：安居客 3800 元（可租）；链家 3900 元（可租）。",
                candidates=[],
                debug={"response_kind": "detail"},
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-3", "message": "HF_4多少钱"},
        )

    assert resp.status_code == 200
    assert _response_text(resp).startswith("HF_4 各平台挂牌")


def test_chat_route_uses_single_llm_pass_for_detail_reply() -> None:
    app = create_app()
    captured: dict = {"calls": []}

    class StubService:
        async def handle(self, request):
            return InvokeResponse(
                text="HF_4：朝阳望京，2居1厅1卫，3800 元/月。离最近地铁约 620 米。当前状态：可租。",
                candidates=[],
                debug={"response_kind": "detail"},
            )

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            captured["calls"].append(json)
            content = json["messages"][0]["content"]
            assert "你是租房智能Agent决策器" in content
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"intent":"chat","tool_plan":{"operationId":"none","arguments":{}},'
                                    '"assistant_reply":"收到，我来帮你确认这套房的关键信息。","confidence":0.7}'
                                )
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-detail-polish", "message": "这套离地铁多远，能租吗？"},
        )

    assert resp.status_code == 200
    assert len(captured["calls"]) == 1
    assert _response_text(resp).startswith("HF_4：朝阳望京")


def test_chat_route_keeps_detail_reply_when_tool_results_exist_with_single_llm_pass() -> None:
    app = create_app()
    captured: dict = {"calls": []}

    class StubService:
        async def handle(self, request):
            record_tool_result(
                name="GET /api/houses/HF_4",
                success=True,
                output={"house_id": "HF_4", "status": "可租", "subway_distance": 620},
                duration_ms=8,
                method="GET",
                url="http://mock/api/houses/HF_4",
                status_code=200,
            )
            return InvokeResponse(
                text="HF_4：朝阳望京，2居1厅1卫，3800 元/月。离最近地铁约 620 米。当前状态：可租。",
                candidates=[],
                debug={"response_kind": "detail"},
            )

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            captured["calls"].append(json)
            assert url == "http://127.0.0.1:8888/v1/chat/completions"
            assert headers["Session-ID"] == "sess-two-llm-detail"
            assert len(captured["calls"]) == 1
            assert isinstance(json["tools"], list) and len(json["tools"]) > 0
            assert "你是租房智能Agent决策器" in json["messages"][0]["content"]
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"intent":"house_detail","tool_plan":{"operationId":"get_house_by_id","arguments":'
                                    '{"house_id":"HF_4"}},"confidence":0.8}'
                                )
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-two-llm-detail", "message": "这套离地铁多远，能租吗？"},
        )

    assert resp.status_code == 200
    assert len(captured["calls"]) == 1
    assert _response_text(resp) == "HF_4：朝阳望京，2居1厅1卫，3800 元/月。离最近地铁约 620 米。当前状态：可租。"


def test_chat_route_llm_nlu_result_is_passed_to_agent_request_meta() -> None:
    app = create_app()
    captured: dict = {}

    class StubState:
        conversation_summary = "上轮推荐 HF_1001"

    class StubStateStore:
        def get(self, session_id):
            assert session_id == "sess-llm-nlu"
            return StubState()

    class StubService:
        state_store = StubStateStore()

        def rough_token_estimate(self, text: str) -> int:
            return 10

        def allow_llm_fallback(self, session_id: str, estimated_prompt_tokens: int) -> bool:
            return True

        def record_llm_fallback_usage(self, session_id: str, consumed_tokens: int) -> None:
            captured["consumed_tokens"] = consumed_tokens

        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="已执行", candidates=[], debug={"response_kind": "action"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert url == "http://127.0.0.1:8888/v1/chat/completions"
            assert headers["Session-ID"] == "sess-llm-nlu"
            assert json["messages"][0]["role"] == "system"
            assert "你是租房智能Agent决策器" in json["messages"][0]["content"]
            assert json["messages"][-1]["role"] == "user"
            user_content = json["messages"][-1]["content"]
            assert "用户输入：帮我把第一套租掉" in user_content
            assert "上轮推荐 HF_1001" in user_content
            assert isinstance(json["tools"], list) and len(json["tools"]) > 0
            tool_names = {item["function"]["name"] for item in json["tools"]}
            assert tool_names == {"landmark", "house_query", "house_action"}
            house_action_tool = next(item for item in json["tools"] if item["function"]["name"] == "house_action")
            action_schema = house_action_tool["function"]["parameters"]
            assert action_schema["required"] == ["action", "house_id", "listing_platform"]
            assert json["stream"] is False
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"intent":"rent","params":{},"tag_need":{"must":[],"avoid":[],"prefer":[]}}',
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-llm-nlu", "message": "帮我把第一套租掉"},
        )

        assert resp.status_code == 200
        assert captured["meta"]["model_ip"] == "127.0.0.1"
        assert captured["meta"]["llm_parse"]["intent"] == "rent"
        assert captured["meta"]["llm_parse"]["params"] == {}


def test_chat_route_llm_tool_call_is_merged_and_tool_params_take_priority() -> None:
    app = create_app()
    captured: dict = {}

    class StubService:
        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "search"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            _ = url
            _ = json
            assert headers["Session-ID"] == "sess-tool-priority"
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "i=search|p=district:海淀;max_price:3000|t=m:;a:;p:",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "house_query",
                                            "arguments": __import__("json").dumps(
                                                {
                                                    "op": "by_platform",
                                                    "district": "朝阳",
                                                    "bedrooms": "2",
                                                    "max_price": 3500,
                                                    "subway_distance": 800,
                                                },
                                                ensure_ascii=False,
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-tool-priority", "message": "帮我找朝阳两居"},
        )

    assert resp.status_code == 200
    llm_parse = captured["meta"]["llm_parse"]
    assert llm_parse["intent"] == "search"
    assert llm_parse["params"]["district"] == "朝阳"
    assert llm_parse["params"]["bedrooms"] == "2"
    assert llm_parse["params"]["max_price"] == 3500
    assert llm_parse["params"]["max_subway_dist"] == 800


def test_chat_route_llm_tool_call_keeps_nearby_max_distance_argument() -> None:
    app = create_app()
    captured: dict = {}

    class StubService:
        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "search"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert headers["Session-ID"] == "sess-nearby-max-distance"
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    '{"intent":"search","params":{"max_subway_dist":500,"bedrooms":"2"},'
                                    '"tag_need":{"must":[],"avoid":[],"prefer":[]}}'
                                ),
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={
                "model_ip": "127.0.0.1",
                "session_id": "sess-nearby-max-distance",
                "message": "车公庄站500米内的两居",
            },
        )

    assert resp.status_code == 200
    assert captured["meta"]["llm_parse"]["intent"] == "search"
    params = captured["meta"]["llm_parse"]["params"]
    assert params["max_subway_dist"] == 500
    assert params["bedrooms"] == "2"


def test_chat_route_llm_tool_call_maps_id_alias_to_house_id_for_house_detail() -> None:
    app = create_app()
    captured: dict = {}

    class StubService:
        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "detail"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert headers["Session-ID"] == "sess-id-alias"
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    '{"intent":"house_detail","params":{"district":"朝阳","house_id":"HF_14"},'
                                    '"tag_need":{"must":[],"avoid":[],"prefer":[]}}'
                                ),
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-id-alias", "message": "看看这个房源详情"},
        )

    assert resp.status_code == 200
    llm_parse = captured["meta"]["llm_parse"]
    assert llm_parse["intent"] == "house_detail"
    assert llm_parse["params"]["district"] == "朝阳"
    assert "house_id" not in llm_parse["params"]


def test_chat_route_merges_llm_content_parse_with_tool_call_plan() -> None:
    app = create_app()
    captured: dict = {}

    class StubService:
        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "search"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert headers["Session-ID"] == "sess-merge-parse"
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"intent":"search","params":{"district":"朝阳","min_price":2600,'
                                    '"max_price":3400,"bedrooms":"2"},"tag_need":{"must":[],"avoid":["年付"],'
                                    '"prefer":["月付","房东直租"]}}'
                                )
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-merge-parse", "message": "朝阳两居三千左右，月付房东直租"},
        )

    assert resp.status_code == 200
    llm_parse = captured["meta"]["llm_parse"]
    assert llm_parse["params"]["min_price"] == 2600
    assert llm_parse["params"]["max_price"] == 3400
    assert llm_parse["params"]["district"] == "朝阳"
    assert "月付" in llm_parse["tag_need"]["prefer"]


def test_chat_route_parses_compact_string_nlu_payload() -> None:
    app = create_app()
    captured: dict = {}

    class StubService:
        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "search"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert headers["Session-ID"] == "sess-compact-nlu"
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "i=search|p=d:朝阳;b:2;min:2600;max:3400;noise_preference:true|"
                                    "t=m:近地铁;a:年付;p:月付,房东直租"
                                )
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-compact-nlu", "message": "朝阳两居三千左右，月付直租"},
        )

    assert resp.status_code == 200
    llm_parse = captured["meta"]["llm_parse"]
    assert llm_parse["intent"] == "search"
    assert llm_parse["params"]["district"] == "朝阳"
    assert llm_parse["params"]["bedrooms"] == "2"
    assert llm_parse["params"]["min_price"] == 2600
    assert llm_parse["params"]["max_price"] == 3400
    assert llm_parse["params"]["noise_preference"] is True
    assert "近地铁" in llm_parse["tag_need"]["must"]
    assert "年付" in llm_parse["tag_need"]["avoid"]
    assert "月付" in llm_parse["tag_need"]["prefer"]
    assert "房东直租" in llm_parse["tag_need"]["prefer"]


def test_chat_route_uses_single_llm_pass_and_persists_top5_context() -> None:
    app = create_app()
    captured: dict = {"llm_calls": 0}

    state = SessionState(session_id="sess-rerank-top5", user_id="u-1", case_type=CaseType.multi)
    state.last_candidates = [
        HouseLite(house_id="HF_1", rent=3200, district="朝阳", layout="2居", subway_distance=900),
        HouseLite(house_id="HF_2", rent=3100, district="朝阳", layout="2居", subway_distance=700),
        HouseLite(house_id="HF_3", rent=3050, district="朝阳", layout="2居", subway_distance=650),
        HouseLite(house_id="HF_4", rent=3300, district="朝阳", layout="2居", subway_distance=800),
        HouseLite(house_id="HF_5", rent=3400, district="朝阳", layout="2居", subway_distance=780),
        HouseLite(house_id="HF_6", rent=2990, district="朝阳", layout="2居", subway_distance=950),
        HouseLite(house_id="HF_7", rent=3000, district="朝阳", layout="2居", subway_distance=500),
        HouseLite(house_id="HF_8", rent=3150, district="朝阳", layout="2居", subway_distance=720),
        HouseLite(house_id="HF_9", rent=3080, district="朝阳", layout="2居", subway_distance=610),
        HouseLite(
            house_id="HF_10",
            rent=3250,
            district="朝阳",
            layout="2居",
            subway_distance=830,
            hidden_noise_level="吵闹",
        ),
    ]
    state.last_top5 = [
        HouseViewModel(house_id="HF_1", rent=3200),
        HouseViewModel(house_id="HF_2", rent=3100),
        HouseViewModel(house_id="HF_3", rent=3050),
        HouseViewModel(house_id="HF_4", rent=3300),
        HouseViewModel(house_id="HF_5", rent=3400),
    ]
    state.search_history = [SearchSnapshot(query_text="朝阳两居", district="朝阳", house_ids=["HF_1", "HF_2", "HF_3", "HF_4", "HF_5"])]
    state.recent_turns = [TurnSummary(user="找房", assistant="初稿", intent=IntentType.search, house_ids=["HF_1", "HF_2", "HF_3"])]

    class StubStateStore:
        def __init__(self):
            self.state = state

        def get(self, session_id):
            assert session_id == "sess-rerank-top5"
            return self.state

        def upsert(self, value):
            self.state = value

    class StubService:
        state_store = StubStateStore()

        def rough_token_estimate(self, text: str) -> int:
            return max(1, int(len(text) / 2))

        def record_llm_fallback_usage(self, session_id: str, consumed_tokens: int) -> None:
            captured["token_used"] = consumed_tokens

        async def handle(self, request):
            return InvokeResponse(
                text="查找到以下符合您要求的房源",
                candidates=[
                    HouseViewModel(house_id="HF_1", rent=3200),
                    HouseViewModel(house_id="HF_2", rent=3100),
                    HouseViewModel(house_id="HF_3", rent=3050),
                    HouseViewModel(house_id="HF_4", rent=3300),
                    HouseViewModel(house_id="HF_5", rent=3400),
                ],
                debug={"response_kind": "search", "intent": "search"},
            )

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            captured["llm_calls"] += 1
            assert headers["Session-ID"] == "sess-rerank-top5"
            assert captured["llm_calls"] == 1
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "i=search|p=d:朝阳;b:2;max:3500|t=m:;a:;p:月付",
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-rerank-top5", "message": "朝阳两居，预算3500，离地铁近"},
        )

    assert resp.status_code == 200
    body = _response_json(resp)
    assert body["message"] == "查找到以下符合您要求的房源"
    assert body["houses"] == ["HF_1", "HF_2", "HF_3", "HF_4", "HF_5"]
    assert state.last_top5[0].house_id == "HF_1"
    assert [item.house_id for item in state.last_top5] == ["HF_1", "HF_2", "HF_3", "HF_4", "HF_5"]
    assert state.candidate_state.latest_house_ids == ["HF_1", "HF_2", "HF_3", "HF_4", "HF_5"]
    assert len(state.house_context_top10) == 10
    assert [item.house_id for item in state.house_context_top10[:5]] == ["HF_1", "HF_2", "HF_3", "HF_4", "HF_5"]
    assert state.search_history[-1].house_ids == ["HF_1", "HF_2", "HF_3", "HF_4", "HF_5"]
    assert captured["llm_calls"] == 1


def test_chat_route_skips_second_llm_rerank_when_candidate_count_lte_5() -> None:
    app = create_app()
    captured: dict = {"llm_calls": 0}

    state = SessionState(session_id="sess-rerank-skip", user_id="u-1", case_type=CaseType.multi)
    state.last_candidates = [
        HouseLite(house_id="HF_A1", rent=3200, district="朝阳", community="朝阳小区A"),
        HouseLite(house_id="HF_A2", rent=3300, district="朝阳", community="朝阳小区B"),
        HouseLite(house_id="HF_A3", rent=3400, district="朝阳", community="朝阳小区C"),
    ]
    state.last_top5 = [
        HouseViewModel(house_id="HF_A1", rent=3200),
        HouseViewModel(house_id="HF_A2", rent=3300),
        HouseViewModel(house_id="HF_A3", rent=3400),
    ]
    state.search_history = [SearchSnapshot(query_text="朝阳两居", district="朝阳", house_ids=["HF_A1", "HF_A2", "HF_A3"])]
    state.recent_turns = [TurnSummary(user="找房", assistant="初稿", intent=IntentType.search, house_ids=["HF_A1"])]

    class StubStateStore:
        def __init__(self):
            self.state = state

        def get(self, session_id):
            assert session_id == "sess-rerank-skip"
            return self.state

        def upsert(self, value):
            self.state = value

    class StubService:
        state_store = StubStateStore()

        async def handle(self, request):
            return InvokeResponse(
                text="已找到3套符合要求的房源",
                candidates=[
                    HouseViewModel(house_id="HF_A1", rent=3200),
                    HouseViewModel(house_id="HF_A2", rent=3300),
                    HouseViewModel(house_id="HF_A3", rent=3400),
                ],
                debug={"response_kind": "search", "intent": "search"},
            )

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            _ = url
            captured["llm_calls"] += 1
            assert headers["Session-ID"] == "sess-rerank-skip"
            if captured["llm_calls"] == 1:
                return StubResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "i=search|p=d:朝阳;b:2;max:3500|t=m:;a:;p:月付",
                                }
                            }
                        ]
                    }
                )
            raise AssertionError("candidate_count <= 5 should not call second llm rerank")

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-rerank-skip", "message": "朝阳两居，预算3500"},
        )

    assert resp.status_code == 200
    body = _response_json(resp)
    assert body["message"] == "已找到3套符合要求的房源"
    assert body["houses"] == ["HF_A1", "HF_A2", "HF_A3"]
    assert len(state.house_context_top10) == 3
    assert captured["llm_calls"] == 1


def test_chat_route_search_context_config_still_works_in_single_llm_mode() -> None:
    app = create_app(
        settings=AgentSettings(
            rerank_house_context_fields="house_id,community,price,subway_distance,status,nearby_landmarks",
            rerank_landmark_context_fields="name,distance,type",
            rerank_landmark_item_limit=1,
        )
    )
    captured: dict = {"llm_calls": 0}

    state = SessionState(session_id="sess-rerank-config", user_id="u-1", case_type=CaseType.multi)
    state.last_candidates = [
        HouseLite(
            house_id="HF_6751",
            community="朝阳小区906",
            district="朝阳",
            rent=3200,
            subway_distance=424,
            status="available",
            listing_platform="安居客",
        ),
        HouseLite(
            house_id="HF_6752",
            community="朝阳小区829",
            district="朝阳",
            rent=3300,
            subway_distance=520,
            status="available",
            listing_platform="安居客",
        ),
        HouseLite(
            house_id="HF_6753",
            community="朝阳小区831",
            district="朝阳",
            rent=3400,
            subway_distance=610,
            status="available",
            listing_platform="安居客",
        ),
        HouseLite(
            house_id="HF_6754",
            community="朝阳小区832",
            district="朝阳",
            rent=3450,
            subway_distance=630,
            status="available",
            listing_platform="安居客",
        ),
        HouseLite(
            house_id="HF_6755",
            community="朝阳小区833",
            district="朝阳",
            rent=3500,
            subway_distance=680,
            status="available",
            listing_platform="安居客",
        ),
        HouseLite(
            house_id="HF_6756",
            community="朝阳小区834",
            district="朝阳",
            rent=3600,
            subway_distance=710,
            status="available",
            listing_platform="安居客",
        ),
    ]
    state.last_top5 = [
        HouseViewModel(house_id="HF_6751", rent=3200, community="朝阳小区906"),
        HouseViewModel(house_id="HF_6752", rent=3300, community="朝阳小区829"),
        HouseViewModel(house_id="HF_6753", rent=3400, community="朝阳小区831"),
        HouseViewModel(house_id="HF_6754", rent=3450, community="朝阳小区832"),
        HouseViewModel(house_id="HF_6755", rent=3500, community="朝阳小区833"),
    ]
    state.search_history = [SearchSnapshot(query_text="朝阳两居", district="朝阳", house_ids=["HF_6751", "HF_6752", "HF_6753"])]
    state.recent_turns = [TurnSummary(user="找房", assistant="初稿", intent=IntentType.search, house_ids=["HF_6751"])]

    class StubStateStore:
        def __init__(self):
            self.state = state

        def get(self, session_id):
            assert session_id == "sess-rerank-config"
            return self.state

        def upsert(self, value):
            self.state = value

    class StubService:
        state_store = StubStateStore()

        async def handle(self, request):
            record_tool_result(
                name="GET /api/houses/nearby_landmarks",
                success=True,
                output={
                    "community": "朝阳小区906",
                    "type": "shopping",
                    "total": 1,
                    "items": [
                        {
                            "landmark": {
                                "id": "LM_004",
                                "name": "王府井步行街",
                                "category": "landmark",
                                "district": "东城",
                                "details": {
                                    "type": "shopping",
                                    "type_name": "商业街",
                                },
                            },
                            "distance": 1888.224411287739,
                        }
                    ],
                },
                duration_ms=7,
                method="GET",
                url="http://mock/api/houses/nearby_landmarks",
                status_code=200,
            )
            return InvokeResponse(
                text="已找到候选房源",
                candidates=[
                    HouseViewModel(house_id="HF_6751", rent=3200, community="朝阳小区906"),
                    HouseViewModel(house_id="HF_6752", rent=3300, community="朝阳小区829"),
                    HouseViewModel(house_id="HF_6753", rent=3400, community="朝阳小区831"),
                    HouseViewModel(house_id="HF_6754", rent=3450, community="朝阳小区832"),
                    HouseViewModel(house_id="HF_6755", rent=3500, community="朝阳小区833"),
                    HouseViewModel(house_id="HF_6756", rent=3600, community="朝阳小区834"),
                ],
                debug={"response_kind": "search", "intent": "search"},
            )

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            _ = url
            captured["llm_calls"] += 1
            assert headers["Session-ID"] == "sess-rerank-config"
            assert captured["llm_calls"] == 1
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "i=search|p=d:朝阳;b:2;max:3500|t=m:;a:;p:月付",
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-rerank-config", "message": "朝阳两居，预算3500"},
        )

    assert resp.status_code == 200
    body = _response_json(resp)
    assert body["message"] == "已找到候选房源"
    assert body["houses"][:2] == ["HF_6751", "HF_6752"]
    assert len(body["houses"]) == 6
    assert len(state.house_context_top10) >= 6
    assert captured["llm_calls"] == 1


def test_chat_route_must_tag_guard_keeps_only_must_matched_houses_in_context() -> None:
    app = create_app()

    state = SessionState(session_id="sess-must-guard", user_id="u-1", case_type=CaseType.multi)
    state.house_context_top10 = [
        HouseLite(house_id="HF_M1", rent=3200, district="朝阳", layout="2居"),
        HouseLite(house_id="HF_N1", rent=3300, district="朝阳", layout="2居"),
    ]
    state.last_candidates = [
        HouseLite(house_id="HF_M1", rent=3200, district="朝阳", layout="2居"),
        HouseLite(house_id="HF_N1", rent=3300, district="朝阳", layout="2居"),
    ]
    state.candidate_state.latest_house_ids = ["HF_M1", "HF_N1"]
    state.candidate_state.focus_house_id = "HF_N1"
    state.focus_house_id = "HF_N1"

    class StubStateStore:
        def __init__(self):
            self.state = state

        def get(self, session_id):
            assert session_id == "sess-must-guard"
            return self.state

        def upsert(self, value):
            self.state = value

    class StubService:
        state_store = StubStateStore()

        async def handle(self, request):
            return InvokeResponse(
                text="已找到候选房源",
                candidates=[
                    HouseViewModel(house_id="HF_N1", rent=3300),
                    HouseViewModel(house_id="HF_M1", rent=3200),
                ],
                debug={
                    "response_kind": "search",
                    "intent": "search",
                    "semantic_filter": {
                        "merged_tag_need": {"must": ["可养狗"], "avoid": [], "prefer": []},
                        "selected": ["HF_M1"],
                        "must_confirm": [{"house_id": "HF_N1", "reason": "刚需标签未明确标注"}],
                        "rejected": [],
                    },
                },
            )

    class StubResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "i=search|p=district:朝阳|t=m:可养狗;a:;p:",
                        }
                    }
                ]
            }

    class StubHttpClient:
        async def post(self, url, json, headers):
            _ = url
            _ = json
            assert headers["Session-ID"] == "sess-must-guard"
            return StubResponse()

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-must-guard", "message": "我要可养狗的两居"},
        )

    assert resp.status_code == 200
    payload = _response_json(resp)
    assert payload["houses"] == ["HF_N1", "HF_M1"]
    assert [item.house_id for item in state.house_context_top10] == ["HF_M1"]
    assert [item.house_id for item in state.last_candidates] == ["HF_M1"]
    assert state.candidate_state.latest_house_ids == ["HF_M1"]
    assert state.candidate_state.focus_house_id == "HF_M1"
    assert state.focus_house_id == "HF_M1"


def test_chat_route_uses_single_llm_call_and_applies_chat_reply_from_nlu() -> None:
    app = create_app()
    captured: dict = {}

    class StubService:
        def rough_token_estimate(self, text: str) -> int:
            return 10

        async def handle(self, request):
            return InvokeResponse(text="规则回复", candidates=[], debug={"response_kind": "chat"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            captured.setdefault("calls", []).append(json)
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"intent":"chat","params":{},"tag_need":{"must":[],"avoid":[],"prefer":[]}}'
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-fallback-content", "message": "用户原话-用于fallback"},
        )

    assert resp.status_code == 200
    assert len(captured["calls"]) == 1
    first_payload = captured["calls"][0]
    assert first_payload["messages"][0]["role"] == "system"
    user_prompt = next(item["content"] for item in first_payload["messages"] if item["role"] == "user")
    assert "用户输入：用户原话-用于fallback" in user_prompt
    assert isinstance(first_payload["tools"], list) and len(first_payload["tools"]) > 0
    assert _response_text(resp) == "规则回复"


def test_chat_route_nlu_prompt_includes_tag_catalog_and_keeps_tag_tokens(tmp_path, monkeypatch) -> None:
    app = create_app()
    captured: dict = {}

    tags_file = tmp_path / "tags.txt"
    tags_file.write_text("可养狗、近公园、月付", encoding="utf-8")
    monkeypatch.setenv("AGENT_TAGS_PATH", str(tags_file))
    _load_nlu_tag_catalog.cache_clear()

    class StubService:
        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="规则回复", candidates=[], debug={"response_kind": "chat"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            _ = url
            _ = headers
            captured["payload"] = json
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "i=search|p=district:朝阳|t=m:;a:;p:可养狗,近公园",
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-tag-catalog", "message": "我想找能养狗且近公园的房子"},
        )

    _load_nlu_tag_catalog.cache_clear()

    assert resp.status_code == 200
    payload = captured["payload"]
    prompt = payload["messages"][0]["content"]
    assert "标签全集：" in prompt
    assert "可养狗" in prompt
    assert "近公园" in prompt
    assert "若同一桶（m/a/p）命中多个标签，必须全部返回" in prompt
    assert "t=m:可养狗,近公园;a:年付,收中介费;p:月付,房东直租" in prompt
    llm_parse = captured["meta"]["llm_parse"]
    assert llm_parse["intent"] == "search"
    assert "可养狗" in llm_parse["tag_need"]["prefer"]
    assert "近公园" in llm_parse["tag_need"]["prefer"]


def test_chat_route_sanitizes_unknown_tool_operation_from_llm() -> None:
    app = create_app()
    captured: dict = {}

    class StubState:
        conversation_summary = "约束[区域=大兴] 焦点[house=HF_4]"
        focus_house_id = "HF_4"
        focus_listing_platform = None
        confirmed_constraints = type(
            "Hard",
            (),
            {"model_dump": staticmethod(lambda exclude_none=True: {"district": "大兴", "budget_max": 4000})},
        )()
        search_history = []
        recent_turns = []

    class StubStateStore:
        def get(self, session_id):
            assert session_id == "sess-unknown-op"
            return StubState()

    class StubService:
        state_store = StubStateStore()

        def rough_token_estimate(self, text: str) -> int:
            return 10

        def allow_llm_fallback(self, session_id: str, estimated_prompt_tokens: int) -> bool:
            return True

        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "chat"})

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert url == "http://127.0.0.1:8888/v1/chat/completions"
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"intent":"search","params":{"district":"大兴","unknown":"x"},'
                                    '"tag_need":{"must":[],"avoid":[],"prefer":[]},"extra":1}'
                                )
                            }
                        }
                    ]
                }
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-unknown-op", "message": "随便查查"},
        )

        assert resp.status_code == 200
        llm_parse = captured["meta"]["llm_parse"]
        assert llm_parse["intent"] == "search"
        assert llm_parse["params"] == {"district": "大兴"}


def test_chat_route_llm_nlu_retries_once_on_read_timeout() -> None:
    app = create_app()
    captured: dict = {"calls": 0}

    class StubService:
        def rough_token_estimate(self, text: str) -> int:
            return 10

        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "action"})

    class StubResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"intent":"search","tool_plan":{"operationId":"get_houses_by_platform","arguments":'
                                '{"district":"朝阳"}},"confidence":0.8}'
                            )
                        }
                    }
                ]
            }

    class StubHttpClient:
        async def post(self, url, json, headers):
            captured["calls"] += 1
            if captured["calls"] == 1:
                raise httpx.ReadTimeout("timeout", request=httpx.Request("POST", url))
            return StubResponse()

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-retry", "message": "帮我查朝阳区"},
        )

        assert resp.status_code == 200
        assert captured["calls"] == 2
        assert captured["meta"]["llm_parse"]["intent"] == "search"


def test_chat_route_llm_nlu_retries_on_model_error_until_third_attempt() -> None:
    app = create_app()
    captured: dict[str, int | dict] = {"calls": 0}

    class StubService:
        def rough_token_estimate(self, text: str) -> int:
            return 10

        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="ok", candidates=[], debug={"response_kind": "action"})

    class StubResponse:
        def __init__(self, status_code: int, payload: dict, url: str):
            self.status_code = status_code
            self._payload = payload
            self._url = url

        def raise_for_status(self):
            if self.status_code < 400:
                return None
            request = httpx.Request("POST", self._url)
            response = httpx.Response(self.status_code, request=request, json=self._payload)
            raise httpx.HTTPStatusError("upstream error", request=request, response=response)

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            captured["calls"] += 1
            if captured["calls"] <= 2:
                return StubResponse(500, {"error": "temporary"}, url)
            return StubResponse(
                200,
                {"choices": [{"message": {"content": "i=search|p=district:朝阳|t=m:;a:;p:"}}]},
                url,
            )

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-retry-3", "message": "帮我查朝阳区"},
        )

        assert resp.status_code == 200
        assert captured["calls"] == 3
        assert captured["meta"]["llm_parse"]["intent"] == "search"


def test_chat_route_always_calls_llm_nlu_even_when_service_policy_disallows() -> None:
    app = create_app()
    captured: dict = {"http_called": False}

    class StubService:
        def should_use_llm_nlu(self, session_id: str, user_message: str):
            return False, "rule_high_conf_search"

        async def handle(self, request):
            captured["meta"] = request.meta
            return InvokeResponse(text="已执行", candidates=[], debug={"response_kind": "action"})

    class StubResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"intent":"search","tool_plan":{"operationId":"get_houses_by_platform","arguments":'
                                '{"district":"海淀"}},"confidence":0.7}'
                            )
                        }
                    }
                ]
            }

    class StubHttpClient:
        async def post(self, url, json, headers):
            captured["http_called"] = True
            assert url == "http://127.0.0.1:8888/v1/chat/completions"
            return StubResponse()

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-skip", "message": "海淀两居预算8000"},
        )

        assert resp.status_code == 200
        assert captured["http_called"] is True
        assert captured["meta"]["model_ip"] == "127.0.0.1"
        assert captured["meta"]["llm_parse"]["intent"] == "search"


def test_proxy_chat_completions_route() -> None:
    app = create_app()
    captured = {"calls": 0}

    class StubResponse:
        def __init__(self, payload, status_code=200, url: str = "http://127.0.0.1:8888/v1/chat/completions"):
            self._payload = payload
            self.status_code = status_code
            self._url = url

        def raise_for_status(self):
            if self.status_code < 400:
                return None
            request = httpx.Request("POST", self._url)
            response = httpx.Response(self.status_code, request=request, json=self._payload)
            raise httpx.HTTPStatusError("upstream error", request=request, response=response)

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            captured["calls"] += 1
            assert url == "http://127.0.0.1:8888/v1/chat/completions"
            assert json["messages"][0]["content"] == "你好"
            assert headers["Session-ID"] == "sess-1"
            if captured["calls"] <= 2:
                return StubResponse({"error": "temporary"}, status_code=500, url=url)
            return StubResponse({"choices": [{"message": {"content": "hi"}}]}, status_code=200, url=url)

    with TestClient(app) as client:
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/vi/chat/completions",
            headers={"Model-IP": "127.0.0.1", "Session-ID": "sess-1"},
            json={"model": "", "messages": [{"role": "user", "content": "你好"}], "stream": False},
        )

        assert resp.status_code == 200
        assert captured["calls"] == 3
        assert resp.json()["choices"][0]["message"]["content"] == "hi"


def test_proxy_chat_completions_use_session_model_ip_when_header_missing() -> None:
    app = create_app()

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "x"}

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert url == "http://127.0.0.1:8888/v1/chat/completions"
            assert headers["Session-ID"] == "sess-2"
            return StubResponse()

    with TestClient(app) as client:
        app.state.http_client = StubHttpClient()
        app.state.session_model_ip["sess-2"] = "127.0.0.1"
        resp = client.post(
            "/vi/chat/completions",
            headers={"Session-ID": "sess-2"},
            json={"model": "", "messages": [{"role": "user", "content": "你好"}], "stream": False},
        )

        assert resp.status_code == 200
        assert resp.json()["id"] == "x"


def test_proxy_chat_completions_requires_model_ip() -> None:
    app = create_app()

    with TestClient(app) as client:
        resp = client.post(
            "/vi/chat/completions",
            json={"model": "", "messages": [{"role": "user", "content": "你好"}], "stream": False},
        )

        assert resp.status_code == 400


def test_proxy_chat_completions_requires_session_id_for_v1() -> None:
    app = create_app()

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            headers={"Model-IP": "127.0.0.1"},
            json={"model": "", "messages": [{"role": "user", "content": "你好"}], "stream": False},
        )

        assert resp.status_code == 400


def test_proxy_chat_completions_v2_without_session_header() -> None:
    app = create_app()

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "v2-ok"}

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert url == "http://127.0.0.1:8888/v1/chat/completions"
            assert headers == {}
            assert json["messages"][0]["content"] == "你好"
            return StubResponse()

    with TestClient(app) as client:
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/v2/chat/completions",
            headers={"Model-IP": "127.0.0.1"},
            json={"model": "", "messages": [{"role": "user", "content": "你好"}], "stream": False},
        )

        assert resp.status_code == 200
        assert resp.json()["id"] == "v2-ok"


def test_http_request_logging(caplog) -> None:
    app = create_app()

    class StubService:
        async def handle(self, request):
            return InvokeResponse(text="ok", candidates=[])

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        with caplog.at_level(logging.INFO):
            resp = client.post(
                "/api/v1/chat",
                json={"model_ip": "127.0.0.1", "session_id": "sess-log", "message": "记录日志"},
            )

    assert resp.status_code == 200
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "\"event\": \"http.request.in\"" in logs
    assert "\"path\": \"/api/v1/chat\"" in logs
    assert "\"event\": \"http.request.out\"" in logs


def test_init_houses_post_does_not_send_empty_json_or_params() -> None:
    captured: dict = {}

    class StubResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class StubAsyncClient:
        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return StubResponse()

    async def run():
        client = HousesClient("http://fake-host:8080", "user-1", StubAsyncClient())
        await client.init_houses()

    asyncio.run(run())

    assert captured["url"] == "http://fake-host:8080/api/houses/init"
    assert "json" not in captured["kwargs"]
    assert "params" not in captured["kwargs"]
    assert captured["kwargs"]["headers"]["X-User-ID"] == "user-1"


def test_init_houses_accepts_non_json_success_response() -> None:
    captured: dict = {}

    class StubResponse:
        status_code = 200
        text = "ok"

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("not json")

    class StubAsyncClient:
        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return StubResponse()

    async def run():
        client = HousesClient("http://fake-host:8080", "user-1", StubAsyncClient())
        return await client.init_houses()

    result = asyncio.run(run())
    assert captured["url"] == "http://fake-host:8080/api/houses/init"
    assert result == {"raw": "ok"}


def test_stage_name_distinguishes_chat_request_and_response() -> None:
    from app.main import _stage_name

    request_stage = _stage_name({"event": "http.agent_io.request", "method": "POST", "path": "/api/v1/chat"})
    response_stage = _stage_name({"event": "http.agent_io.response", "method": "POST", "path": "/api/v1/chat"})

    assert request_stage == "agent接收用户输入"
    assert response_stage == "agent最终返回用户"


def test_debug_agent_io_events_support_session_filter(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "agent_http_io.log"
    log_file.write_text(
        "\n".join(
            [
                '2026-03-03 10:00:00 {"event":"http.agent_io.request","method":"POST","path":"/api/v1/chat","session_id":"sess-a"}',
                '2026-03-03 10:00:01 {"event":"http.agent_io.api.request","method":"GET","url":"http://x/api/houses/by_platform","session_id":"sess-a"}',
                '2026-03-03 10:00:02 {"event":"http.agent_io.request","method":"GET","path":"/debug/agent-io/events","session_id":"-"}',
                '2026-03-03 10:00:03 {"event":"http.agent_io.response","method":"POST","path":"/api/v1/chat","session_id":"sess-b"}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_HTTP_IO_LOG_PATH", str(log_file))

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/debug/agent-io/events", params={"session_id": "sess-a", "limit": 50})

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sess-a"
    assert body["count"] == 2
    assert all(item["session_id"] == "sess-a" for item in body["items"])


def test_debug_agent_io_events_decode_nested_json_strings_for_display(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "agent_http_io.log"
    payload = {
        "event": "http.agent_io.llm.request",
        "session_id": "sess-x",
        "method": "POST",
        "path": "/api/v1/chat",
        "request_body": '{"model":"","messages":[{"role":"user","content":"hi"}],"tools":[]}',
        "response_body": '{"choices":[{"message":{"content":"ok"}}]}',
    }
    log_file.write_text(
        "2026-03-03 10:00:00 " + json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_HTTP_IO_LOG_PATH", str(log_file))

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/debug/agent-io/events", params={"session_id": "sess-x", "limit": 20})

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    first = body["items"][0]
    assert isinstance(first.get("request_body"), dict)
    assert isinstance(first.get("response_body"), dict)
    assert first["request_body"]["messages"][0]["role"] == "user"
    assert first["response_body"]["choices"][0]["message"]["content"] == "ok"


def test_debug_agent_io_events_keep_large_body_without_truncation(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "agent_http_io.log"
    large_prompt = "x" * 12000
    payload = {
        "event": "http.agent_io.llm.request",
        "session_id": "sess-big",
        "method": "POST",
        "path": "/api/v1/chat",
        "request_body": {
            "model": "",
            "messages": [{"role": "user", "content": large_prompt}],
            "tools": [],
            "stream": False,
        },
    }
    log_file.write_text(
        "2026-03-03 10:00:00 " + json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_HTTP_IO_LOG_PATH", str(log_file))

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/debug/agent-io/events", params={"session_id": "sess-big", "limit": 20})

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    first = body["items"][0]
    content = first["request_body"]["messages"][0]["content"]
    assert content == large_prompt
    assert "...(truncated)" not in content


def test_debug_agent_io_focus_events_only_keep_user_and_llm(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "agent_http_io.log"
    lines = [
        {
            "event": "http.agent_io.request",
            "method": "POST",
            "path": "/api/v1/chat",
            "session_id": "sess-focus",
            "request_body": {"session_id": "sess-focus", "model_ip": "127.0.0.1", "message": "帮我找安静的两居"},
        },
        {
            "event": "http.agent_io.api.request",
            "method": "GET",
            "url": "http://x/api/houses/by_platform",
            "session_id": "sess-focus",
        },
        {
            "event": "http.agent_io.llm.request",
            "method": "POST",
            "url": "http://127.0.0.1:8000/v1/chat/completions",
            "session_id": "sess-focus",
            "llm_stage": "plan",
            "request_body": {
                "model": "",
                "messages": [{"role": "user", "content": "安静两居"}],
                "tools": [
                    {"type": "function", "function": {"name": "house_query"}},
                    {"type": "function", "function": {"name": "landmark"}},
                ],
                "stream": False,
            },
        },
        {
            "event": "http.agent_io.llm.response",
            "method": "POST",
            "url": "http://127.0.0.1:8000/v1/chat/completions",
            "session_id": "sess-focus",
            "llm_stage": "plan",
            "status_code": 200,
            "response_body": {
                "choices": [{"message": {"content": "i=search|p=noise_preference:true|t=m:;a:;p:"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        },
        {
            "event": "http.agent_io.response",
            "method": "POST",
            "path": "/api/v1/chat",
            "session_id": "sess-focus",
            "response_body": {"session_id": "sess-focus", "response": "{\"message\":\"ok\",\"houses\":[]}"},
        },
    ]
    raw = "\n".join(f"2026-03-03 10:00:0{idx} {json.dumps(item, ensure_ascii=False)}" for idx, item in enumerate(lines))
    log_file.write_text(raw + "\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_HTTP_IO_LOG_PATH", str(log_file))

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/debug/agent-io/focus/events", params={"session_id": "sess-focus", "limit": 50})

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sess-focus"
    assert body["count"] == 4
    events = [item["event"] for item in body["items"]]
    assert "http.agent_io.api.request" not in events
    assert events == [
        "http.agent_io.request",
        "http.agent_io.llm.request",
        "http.agent_io.llm.response",
        "http.agent_io.response",
    ]

    llm_input = next(item for item in body["items"] if item["event"] == "http.agent_io.llm.request")
    assert llm_input["llm_input"]["messages"][0]["content"] == "安静两居"
    assert llm_input["llm_input"]["tools_count"] == 2
    assert llm_input["llm_input"]["tool_names"] == ["house_query", "landmark"]
    assert "tools" not in llm_input["llm_input"]


def test_debug_agent_io_focus_page_renders(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "agent_http_io.log"
    log_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("AGENT_HTTP_IO_LOG_PATH", str(log_file))

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/debug/agent-io/focus")

    assert resp.status_code == 200
    assert "/debug/agent-io/focus/events" in resp.text
