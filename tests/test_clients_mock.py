import httpx
import pytest

from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient


@pytest.mark.asyncio
async def test_houses_client_injects_user_header() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-User-ID") == "u-123"
        return httpx.Response(200, json={"data": {"items": [], "total": 0, "page_size": 10}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = HousesClient("http://test", "u-123", http_client)
        resp = await client.by_platform(page=1, page_size=10)
        assert resp["total"] == 0


@pytest.mark.asyncio
async def test_landmarks_client_without_user_header_requirement() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "X-User-ID" not in request.headers
        return httpx.Response(200, json={"data": [{"id": "L1", "name": "西二旗", "category": "subway"}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = LandmarksClient("http://test", "u-123", http_client)
        rows = await client.list_landmarks()
        assert rows[0].name == "西二旗"


@pytest.mark.asyncio
async def test_houses_client_normalizes_area_to_float() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [
                        {
                            "house_id": "HF_2001",
                            "area": "89.5㎡",
                            "rent": 7600,
                        }
                    ],
                    "total": 1,
                    "page_size": 10,
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = HousesClient("http://test", "u-123", http_client)
        resp = await client.by_platform(page=1, page_size=10)
        assert resp["items"][0].area == 89.5
