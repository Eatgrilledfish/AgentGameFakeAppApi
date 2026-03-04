from __future__ import annotations

import re
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


def _normalize_decoration_param(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned in {"精装", "精装修"}:
        return "精装"
    if cleaned in {"简装", "简装修"}:
        return "简装"
    return None


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
    ) -> dict[str, Any]:
        params = {
            "community": community,
            "listing_platform": listing_platform,
            "page": page,
            "page_size": page_size,
        }
        data = await self._get("/api/houses/by_community", params=_clean_params(params), need_user_id=True)
        normalized = _build_page(data)
        normalized["items"] = [HouseLite.model_validate(_normalize_house(item)) for item in normalized["items"]]
        return normalized

    async def by_platform(
        self,
        listing_platform: str | None = None,
        district: str | None = None,
        area: str | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        bedrooms: str | None = None,
        rental_type: str | None = None,
        decoration: str | None = None,
        orientation: str | None = None,
        elevator: str | None = None,
        min_area: int | None = None,
        max_area: int | None = None,
        property_type: str | None = None,
        subway_line: str | None = None,
        subway_distance: int | None = None,
        max_subway_dist: int | None = None,
        subway_station: str | None = None,
        utilities_type: str | None = None,
        available_from_before: str | None = None,
        commute_to_xierqi_max: int | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        params = {
            "listing_platform": listing_platform,
            "district": district,
            "area": area,
            "min_price": min_price,
            "max_price": max_price,
            "bedrooms": bedrooms,
            "rental_type": rental_type,
            "decoration": _normalize_decoration_param(decoration),
            "orientation": orientation,
            "elevator": elevator,
            "min_area": min_area,
            "max_area": max_area,
            "property_type": property_type,
            "subway_line": subway_line,
            "subway_distance": subway_distance if subway_distance is not None else max_subway_dist,
            "subway_station": subway_station,
            "utilities_type": utilities_type,
            "available_from_before": available_from_before,
            "commute_to_xierqi_max": commute_to_xierqi_max,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "page": page,
            "page_size": page_size,
        }
        data = await self._get("/api/houses/by_platform", params=_clean_params(params), need_user_id=True)
        normalized = _build_page(data)
        normalized["items"] = [HouseLite.model_validate(_normalize_house(item)) for item in normalized["items"]]
        return normalized

    async def nearby(
        self,
        landmark_id: str,
        max_distance: float | int = 2000,
        listing_platform: str | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        params = {
            "landmark_id": landmark_id,
            "max_distance": max_distance,
            "listing_platform": listing_platform,
            "page": page,
            "page_size": page_size,
        }
        data = await self._get("/api/houses/nearby", params=_clean_params(params), need_user_id=True)
        normalized = _build_page(data)
        normalized["items"] = [HouseLite.model_validate(_normalize_house(item)) for item in normalized["items"]]
        return normalized

    async def nearby_landmarks(
        self,
        community: str,
        category: str,
        max_distance_m: float | int = 3000,
    ) -> list[NearbyLandmark]:
        params = {"community": community, "type": category, "max_distance_m": max_distance_m}
        data = await self._get("/api/houses/nearby_landmarks", params=params, need_user_id=True)
        items = _build_page(data)["items"]
        if not items and isinstance(data, list):
            items = data
        normalized = [_normalize_nearby_landmark(item, fallback_category=category) for item in items if isinstance(item, dict)]
        return [NearbyLandmark.model_validate(item) for item in normalized if item]

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

    _normalize_area_fields(normalized)
    return normalized


def _normalize_area_fields(normalized: dict[str, Any]) -> None:
    if "business_area" in normalized and isinstance(normalized["business_area"], str):
        return

    for key in ("biz_area", "area_name", "trade_area"):
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            normalized["business_area"] = value.strip()
            break

    area_value = normalized.get("area")
    parsed_area = _coerce_area_to_float(area_value)
    if parsed_area is not None:
        normalized["area"] = parsed_area
        return

    if isinstance(area_value, str) and area_value.strip():
        normalized["business_area"] = area_value.strip()
        normalized["area"] = None

    for key in ("area_sqm", "house_area", "size", "usable_area", "building_area"):
        parsed = _coerce_area_to_float(normalized.get(key))
        if parsed is not None:
            normalized["area"] = parsed
            return


def _coerce_area_to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None

    if re.fullmatch(r"\d+(?:\.\d+)?", stripped):
        return float(stripped)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:平米|平方米|㎡|m2|M2)", stripped)
    if m:
        return float(m.group(1))
    return None


def _normalize_nearby_landmark(raw: dict[str, Any], *, fallback_category: str | None = None) -> dict[str, Any]:
    # API may return either flat {name, category, distance_m} or nested
    # {"landmark": {...}, "distance": ...}.
    item = dict(raw)
    landmark = item.get("landmark")
    if isinstance(landmark, dict):
        item.setdefault("name", landmark.get("name"))
        item.setdefault("category", landmark.get("category"))

    if "distance_m" not in item and "distance" in item:
        item["distance_m"] = item.get("distance")
    if "category" not in item or not isinstance(item.get("category"), str):
        if isinstance(fallback_category, str) and fallback_category:
            item["category"] = fallback_category
    return item
