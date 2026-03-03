import httpx
import pytest

from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.infra.tool_recorder import begin_tool_recording, get_tool_results, reset_tool_recording


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
        assert "area=%E8%A5%BF%E4%BA%8C%E6%97%97%2C%E4%B8%8A%E5%9C%B0" in request.url.query.decode()
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
async def test_by_platform_normalizes_decoration_to_upstream_allowed_values() -> None:
    seen: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("decoration"))
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [],
                    "total": 0,
                    "page_size": 10,
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = HousesClient("http://test", "u-123", http_client)
        await client.by_platform(decoration="精装修", page=1, page_size=10)
        await client.by_platform(decoration="豪华", page=1, page_size=10)

    assert seen[0] == "精装"
    assert seen[1] is None


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


@pytest.mark.asyncio
async def test_base_client_records_upstream_output_into_tool_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": {"items": [{"house_id": "HF_9"}], "total": 1, "page_size": 10}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        token = begin_tool_recording()
        try:
            client = HousesClient("http://test", "u-123", http_client)
            await client.by_platform(page=1, page_size=10)
            tool_results = get_tool_results()
        finally:
            reset_tool_recording(token)

    assert len(tool_results) == 1
    first = tool_results[0]
    assert first["name"] == "GET /api/houses/by_platform"
    assert first["success"] is True
    assert first["output"]["items"][0]["house_id"] == "HF_9"


@pytest.mark.asyncio
async def test_nearby_landmarks_normalizes_nested_distance_shape() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/houses/nearby_landmarks")
        return httpx.Response(
            200,
            json={
                "data": {
                    "community": "测试小区",
                    "type": "park",
                    "total": 1,
                    "items": [
                        {
                            "landmark": {"name": "奥林匹克公园", "category": "landmark"},
                            "distance": 822.5,
                        }
                    ],
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = HousesClient("http://test", "u-123", http_client)
        rows = await client.nearby_landmarks("测试小区", "park")

    assert len(rows) == 1
    assert rows[0].name == "奥林匹克公园"
    assert rows[0].distance_m == 822.5
