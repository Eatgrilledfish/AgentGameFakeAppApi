from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random

from app.clients.exceptions import DataSourceError
from app.infra.logging import log_event, preview_payload

LOGGER = logging.getLogger(__name__)
STEP_UPSTREAM_API = "STEP-03-UPSTREAM-API"


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

    @staticmethod
    def _encode_query_params(params: dict[str, Any] | None) -> str:
        if not params:
            return ""

        encoded_pairs: list[str] = []
        for key, value in params.items():
            if value is None:
                continue
            encoded_key = quote(str(key), safe="")
            if isinstance(value, str):
                encoded_value = quote(value, safe="")
            else:
                encoded_value = quote(str(value), safe="")
            encoded_pairs.append(f"{encoded_key}={encoded_value}")
        return "&".join(encoded_pairs)

    def _build_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        query = self._encode_query_params(params)
        if not query:
            return url
        return f"{url}?{query}"

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        need_user_id: bool = False,
    ) -> Any:
        url = self._build_url(path, params)
        headers = self._headers_houses() if need_user_id else None

        log_event(
            LOGGER,
            "upstream.request",
            step=STEP_UPSTREAM_API,
            method="GET",
            url=url,
            params=params if params is not None else {},
            headers=self._sanitize_headers(headers),
        )

        async def request() -> httpx.Response:
            return await self.http_client.get(url, headers=headers)

        try:
            response = await self._retry_get(request)
            response.raise_for_status()
            payload = self._unwrap(response.json())
            log_event(
                LOGGER,
                "upstream.response",
                step=STEP_UPSTREAM_API,
                method="GET",
                url=url,
                status_code=response.status_code,
                body=preview_payload(payload),
            )
            return payload
        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
            log_event(LOGGER, "upstream.error", step=STEP_UPSTREAM_API, method="GET", url=url, error=str(exc))
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
        url = self._build_url(path, params)
        headers = self._headers_houses() if need_user_id else None
        log_event(
            LOGGER,
            "upstream.request",
            step=STEP_UPSTREAM_API,
            method="POST",
            url=url,
            params=params if params is not None else {},
            json=json if json is not None else None,
            headers=self._sanitize_headers(headers),
        )
        try:
            kwargs: dict[str, Any] = {"headers": headers}
            if json is not None:
                kwargs["json"] = json
            response = await self.http_client.post(url, **kwargs)
            response.raise_for_status()
            payload: Any
            try:
                payload = self._unwrap(response.json())
            except ValueError:
                # Some endpoints may return empty/plain-text success bodies.
                text_body = response.text.strip() if isinstance(response.text, str) else ""
                payload = {"raw": text_body} if text_body else {}
            log_event(
                LOGGER,
                "upstream.response",
                step=STEP_UPSTREAM_API,
                method="POST",
                url=url,
                status_code=response.status_code,
                body=preview_payload(payload),
            )
            return payload
        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
            log_event(LOGGER, "upstream.error", step=STEP_UPSTREAM_API, method="POST", url=url, error=str(exc))
            LOGGER.warning("POST %s failed: %s", url, exc)
            raise DataSourceError(f"POST failed: {url}") from exc
