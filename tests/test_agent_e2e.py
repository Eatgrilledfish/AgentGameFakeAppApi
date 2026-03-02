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
    assert "incoming request method=POST path=/api/v1/chat" in logs
    assert "request completed method=POST path=/api/v1/chat status=200" in logs


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
