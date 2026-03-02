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
async def test_by_platform_area_query_is_business_area_string() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("area") == "西二旗,上地"
        assert request.url.params.get("min_area") == "60"
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [{"house_id": "HF_1", "area": "西二旗", "rent": 6200, "status": "可租"}],
                    "total": 1,
                    "page_size": 10,
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = HousesClient("http://test", "u-123", http_client)
        resp = await client.by_platform(area="西二旗,上地", min_area=60, page=1, page_size=10)
        assert resp["items"][0].area is None
        assert resp["items"][0].business_area == "西二旗"


@pytest.mark.asyncio
async def test_house_area_string_can_be_parsed_to_float() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [{"house_id": "HF_2", "area": "66.5平米", "rent": 7200, "status": "可租"}],
                    "total": 1,
                    "page_size": 10,
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = HousesClient("http://test", "u-123", http_client)
        resp = await client.by_platform(page=1, page_size=10)
        assert resp["items"][0].area == 66.5


@pytest.mark.asyncio
async def test_rent_post_uses_query_param_without_json_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers.get("X-User-ID") == "u-123"
        assert request.url.params.get("listing_platform") == "链家"
        assert request.content in (b"", None)
        return httpx.Response(200, json={"data": {"ok": True}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = HousesClient("http://test", "u-123", http_client)
        resp = await client.rent("HF_2001", "链家")
        assert resp["ok"] is True


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
