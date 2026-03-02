from __future__ import annotations

from typing import Any

import httpx

from app.clients.base import BaseClient
from app.schemas import Landmark


class LandmarksClient(BaseClient):
    def __init__(self, base_url: str, user_id: str, http_client: httpx.AsyncClient) -> None:
        super().__init__(base_url, user_id, http_client)

    async def list_landmarks(self, category: str | None = None, district: str | None = None) -> list[Landmark]:
        params = {"category": category, "district": district}
        data = await self._get("/api/landmarks", params={k: v for k, v in params.items() if v is not None})
        return [Landmark.model_validate(item) for item in _items(data)]

    async def get_by_name(self, name: str) -> Landmark | None:
        data = await self._get(f"/api/landmarks/name/{name}")
        if not data:
            return None
        return Landmark.model_validate(data)

    async def search(
        self, keyword: str, category: str | None = None, district: str | None = None
    ) -> list[Landmark]:
        params = {"q": keyword, "category": category, "district": district}
        data = await self._get("/api/landmarks/search", params={k: v for k, v in params.items() if v is not None})
        return [Landmark.model_validate(item) for item in _items(data)]

    async def get_detail(self, landmark_id: str) -> Landmark | None:
        data = await self._get(f"/api/landmarks/{landmark_id}")
        if not data:
            return None
        return Landmark.model_validate(data)

    async def stats(self) -> dict[str, Any]:
        data = await self._get("/api/landmarks/stats")
        return data if isinstance(data, dict) else {"value": data}


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []
