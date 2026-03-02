from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random

from app.clients.exceptions import DataSourceError

LOGGER = logging.getLogger(__name__)


class BaseClient:
    def __init__(self, base_url: str, user_id: str, http_client: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.http_client = http_client

    def _headers_houses(self) -> dict[str, str]:
        return {"X-User-ID": self.user_id}

    @staticmethod
    def _unwrap(resp_json: Any) -> Any:
        if isinstance(resp_json, dict) and "data" in resp_json:
            return resp_json["data"]
        return resp_json

    async def _retry_get(self, fn: Callable[[], Any]) -> Any:
        retry = AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_random(min=0.0, max=0.05),
            reraise=True,
            retry=retry_if_exception_type((httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout)),
        )
        async for attempt in retry:
            with attempt:
                return await fn()
        raise RuntimeError("unreachable")

    @staticmethod
    def _sanitize_headers(headers: dict[str, str] | None) -> dict[str, str]:
        if not headers:
            return {}
        return {k: ("***" if "authorization" in k.lower() else v) for k, v in headers.items()}

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        need_user_id: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = self._headers_houses() if need_user_id else None

        LOGGER.info("outgoing request method=GET url=%s params=%s headers=%s", url, params or {}, self._sanitize_headers(headers))

        async def request() -> httpx.Response:
            return await self.http_client.get(url, params=params, headers=headers)

        try:
            response = await self._retry_get(request)
            response.raise_for_status()
            LOGGER.info("outgoing response method=GET url=%s status=%s", url, response.status_code)
            return self._unwrap(response.json())
        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
            LOGGER.warning("GET %s failed: %s", url, exc)
            raise DataSourceError(f"GET failed: {url}") from exc

    async def _post(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        need_user_id: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = self._headers_houses() if need_user_id else None
        LOGGER.info(
            "outgoing request method=POST url=%s params=%s json=%s headers=%s",
            url,
            params or {},
            json or {},
            self._sanitize_headers(headers),
        )
        try:
            response = await self.http_client.post(url, params=params, json=json, headers=headers)
            response.raise_for_status()
            LOGGER.info("outgoing response method=POST url=%s status=%s", url, response.status_code)
            return self._unwrap(response.json())
        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
            LOGGER.warning("POST %s failed: %s", url, exc)
            raise DataSourceError(f"POST failed: {url}") from exc
