import logging
import asyncio
from fastapi.testclient import TestClient

import httpx

from app.clients.houses import HousesClient
from app.main import create_app
from app.schemas import InvokeResponse


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
        assert body["response"]["message"] == "给你筛选好了"
        assert body["response"]["houses"] == ["HF_2101"]
        assert body["status"] == "success"
        assert body["timestamp"] < 10_000_000_000


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
        payload = resp.json()["response"]
        assert payload == {"message": "当前条件无结果", "houses": []}


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
        assert resp.json()["response"]["message"].startswith("HF_4 各平台挂牌")
        assert "houses" not in resp.json()["response"]


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
            assert "意图解析器" in json["messages"][0]["content"]
            assert "operationId" in json["messages"][0]["content"]
            return StubResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"intent":"rent","tool_plan":{"operationId":"rent_house","arguments":'
                                    '{"house_id":"HF_1001","listing_platform":"安居客"}},"confidence":0.92}'
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
            json={"model_ip": "127.0.0.1", "session_id": "sess-llm-nlu", "message": "帮我把第一套租掉"},
        )

        assert resp.status_code == 200
        assert captured["meta"]["model_ip"] == "127.0.0.1"
        assert captured["meta"]["llm_parse"]["intent"] == "rent"
        assert captured["meta"]["llm_parse"]["tool_plan"]["arguments"]["house_id"] == "HF_1001"


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
                                    '{"intent":"search","tool_plan":{"operationId":"fake_op","arguments":'
                                    '{"x":"y","house_id":"BAD-ID"}},"confidence":0.91}'
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
        tool_plan = captured["meta"]["llm_parse"]["tool_plan"]
        assert tool_plan["operationId"] == "none"
        assert tool_plan["arguments"] == {}


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

    class StubResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class StubHttpClient:
        async def post(self, url, json, headers):
            assert url == "http://127.0.0.1:8888/v1/chat/completions"
            assert json["messages"][0]["content"] == "你好"
            assert headers["Session-ID"] == "sess-1"
            return StubResponse({"choices": [{"message": {"content": "hi"}}]})

    with TestClient(app) as client:
        app.state.http_client = StubHttpClient()
        resp = client.post(
            "/vi/chat/completions",
            headers={"Model-IP": "127.0.0.1", "Session-ID": "sess-1"},
            json={"model": "", "messages": [{"role": "user", "content": "你好"}], "stream": False},
        )

        assert resp.status_code == 200
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
