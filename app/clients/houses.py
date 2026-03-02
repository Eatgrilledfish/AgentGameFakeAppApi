from __future__ import annotations

from typing import Any

import httpx

from app.clients.base import BaseClient
from app.schemas import HouseLite, Listing, NearbyLandmark


def _build_page(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return {
                "items": [item for item in items if isinstance(item, dict)],
                "total": int(data.get("total", len(items))),
                "page_size": int(data.get("page_size", len(items) or 10)),
            }
    if isinstance(data, list):
        return {"items": [item for item in data if isinstance(item, dict)], "total": len(data), "page_size": len(data)}
    return {"items": [], "total": 0, "page_size": 10}


class HousesClient(BaseClient):
    def __init__(self, base_url: str, user_id: str, http_client: httpx.AsyncClient) -> None:
        super().__init__(base_url, user_id, http_client)

    async def init_houses(self) -> dict[str, Any]:
        data = await self._post("/api/houses/init", need_user_id=True)
        return data if isinstance(data, dict) else {"value": data}

    async def get_house_detail(self, house_id: str) -> HouseLite | None:
        data = await self._get(f"/api/houses/{house_id}", need_user_id=True)
        if not isinstance(data, dict):
            return None
        return HouseLite.model_validate(_normalize_house(data))

    async def get_listings(self, house_id: str) -> dict[str, Any]:
        data = await self._get(f"/api/houses/listings/{house_id}", need_user_id=True)
        page = _build_page(data)
        page["items"] = [Listing.model_validate(item) for item in page["items"]]
        return page

    async def by_community(
        self,
        community: str,
        listing_platform: str | None = None,
        page: int = 1,
        page_size: int = 10,
        **filters: Any,
    ) -> dict[str, Any]:
        params = {
            "community": community,
            "listing_platform": listing_platform,
            "page": page,
            "page_size": page_size,
            **filters,
        }
        data = await self._get("/api/houses/by_community", params=_clean_params(params), need_user_id=True)
        normalized = _build_page(data)
        normalized["items"] = [HouseLite.model_validate(_normalize_house(item)) for item in normalized["items"]]
        return normalized

    async def by_platform(
        self,
        listing_platform: str | None = None,
        page: int = 1,
        page_size: int = 10,
        **filters: Any,
    ) -> dict[str, Any]:
        params = {
            "listing_platform": listing_platform,
            "page": page,
            "page_size": page_size,
            **filters,
        }
        data = await self._get("/api/houses/by_platform", params=_clean_params(params), need_user_id=True)
        normalized = _build_page(data)
        normalized["items"] = [HouseLite.model_validate(_normalize_house(item)) for item in normalized["items"]]
        return normalized

    async def nearby(
        self,
        landmark_id: str,
        max_distance: int = 2000,
        listing_platform: str | None = None,
        page: int = 1,
        page_size: int = 10,
        **filters: Any,
    ) -> dict[str, Any]:
        params = {
            "landmark_id": landmark_id,
            "max_distance": max_distance,
            "listing_platform": listing_platform,
            "page": page,
            "page_size": page_size,
            **filters,
        }
        data = await self._get("/api/houses/nearby", params=_clean_params(params), need_user_id=True)
        normalized = _build_page(data)
        normalized["items"] = [HouseLite.model_validate(_normalize_house(item)) for item in normalized["items"]]
        return normalized

    async def nearby_landmarks(
        self,
        community: str,
        category: str,
        max_distance_m: int = 3000,
    ) -> list[NearbyLandmark]:
        params = {"community": community, "type": category, "max_distance_m": max_distance_m}
        data = await self._get("/api/houses/nearby_landmarks", params=params, need_user_id=True)
        items = _build_page(data)["items"]
        if not items and isinstance(data, list):
            items = data
        return [NearbyLandmark.model_validate(item) for item in items if isinstance(item, dict)]

    async def stats(self) -> dict[str, Any]:
        data = await self._get("/api/houses/stats", need_user_id=True)
        return data if isinstance(data, dict) else {"value": data}

    async def rent(self, house_id: str, listing_platform: str) -> dict[str, Any]:
        data = await self._post(
            f"/api/houses/{house_id}/rent",
            params={"listing_platform": listing_platform},
            need_user_id=True,
        )
        return data if isinstance(data, dict) else {"value": data}

    async def terminate(self, house_id: str, listing_platform: str) -> dict[str, Any]:
        data = await self._post(
            f"/api/houses/{house_id}/terminate",
            params={"listing_platform": listing_platform},
            need_user_id=True,
        )
        return data if isinstance(data, dict) else {"value": data}

    async def offline(self, house_id: str, listing_platform: str) -> dict[str, Any]:
        data = await self._post(
            f"/api/houses/{house_id}/offline",
            params={"listing_platform": listing_platform},
            need_user_id=True,
        )
        return data if isinstance(data, dict) else {"value": data}


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


def _normalize_house(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    house_id = normalized.get("house_id") or normalized.get("id")
    if house_id is not None:
        normalized["house_id"] = str(house_id)

    if "commute_to_xierqi_min" not in normalized:
        for field in ("commute_to_xierqi", "commute_time_to_xierqi", "xierqi_commute_min"):
            if field in normalized:
                normalized["commute_to_xierqi_min"] = normalized.get(field)
                break

    if "rent" not in normalized:
        if "price" in normalized:
            normalized["rent"] = normalized.get("price")

    if "layout" not in normalized and "house_layout" in normalized:
        normalized["layout"] = normalized.get("house_layout")

    if "status" not in normalized and "house_status" in normalized:
        normalized["status"] = normalized.get("house_status")

    return normalized
