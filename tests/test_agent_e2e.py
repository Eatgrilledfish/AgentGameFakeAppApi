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
        assert "HF_2101" in body["response"]
        assert body["status"] == "success"


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


def test_debug_trace_viewer_endpoints() -> None:
    app = create_app()

    class StubService:
        async def handle(self, request):
            return InvokeResponse(text="ok", candidates=[])

        def rough_token_estimate(self, text: str) -> int:
            return 1

        def allow_llm_fallback(self, session_id: str, estimated_prompt_tokens: int) -> bool:
            return False

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        page = client.get("/debug/chat-view")
        assert page.status_code == 200
        assert "实时交互可视化面板" in page.text

        chat_resp = client.post(
            "/api/v1/chat",
            json={"model_ip": "127.0.0.1", "session_id": "sess-debug", "message": "看日志"},
        )
        assert chat_resp.status_code == 200

        trace_resp = client.get("/debug/traces/sess-debug")
        assert trace_resp.status_code == 200
        events = trace_resp.json()["events"]
        assert any(item.get("event") == "chat.received" for item in events)
        assert any(item.get("event") == "chat.completed" for item in events)


        trace_all = client.get("/debug/traces")
        assert trace_all.status_code == 200
        all_events = trace_all.json()["events"]
        assert any(item.get("session_id") == "sess-debug" for item in all_events)

        max_eid = max(int(item.get("eid", 0)) for item in all_events)
        trace_delta_resp = client.get(f"/debug/traces?since_eid={max_eid - 1}")
        assert trace_delta_resp.status_code == 200
        delta_events = trace_delta_resp.json()["events"]
        assert all(int(item.get("eid", 0)) > max_eid - 1 for item in delta_events)

        clear_resp = client.delete("/debug/traces/sess-debug")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["cleared"] >= 1


def test_debug_trace_viewer_listens_all_sessions() -> None:
    app = create_app()

    class StubService:
        async def handle(self, request):
            return InvokeResponse(text="ok", candidates=[])

        def rough_token_estimate(self, text: str) -> int:
            return 1

        def allow_llm_fallback(self, session_id: str, estimated_prompt_tokens: int) -> bool:
            return False

    with TestClient(app) as client:
        app.state.agent_service = StubService()
        resp1 = client.post("/api/v1/chat", json={"model_ip": "127.0.0.1", "session_id": "sess-a", "message": "A"})
        resp2 = client.post("/api/v1/chat", json={"model_ip": "127.0.0.1", "session_id": "sess-b", "message": "B"})
        assert resp1.status_code == 200
        assert resp2.status_code == 200

        trace_all = client.get("/debug/traces")
        assert trace_all.status_code == 200
        sessions = {item.get("session_id") for item in trace_all.json()["events"]}
        assert "sess-a" in sessions
        assert "sess-b" in sessions
