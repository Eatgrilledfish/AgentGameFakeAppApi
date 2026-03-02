from fastapi.testclient import TestClient

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
