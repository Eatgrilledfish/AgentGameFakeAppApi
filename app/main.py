from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from app.agent.service import AgentService
from app.agent.state import StateStore
from app.clients.exceptions import DataSourceError
from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.infra.cache import CacheManager
from app.infra.logging import (
    bind_log_context,
    get_log_context,
    log_event,
    log_json_event,
    preview_payload,
    preview_text,
    reset_log_context,
    setup_logging,
)
from app.infra.tool_recorder import begin_tool_recording, get_tool_results, reset_tool_recording
from app.schemas import CaseType, ChatRequest, ChatResponse, HealthResponse, HouseLite, HouseViewModel, InvokeRequest, InvokeResponse
from app.settings import AgentSettings, load_settings

LOGGER = logging.getLogger(__name__)
HTTP_IO_LOGGER = logging.getLogger("agent.http.io")

STEP_RECV_USER_QUERY = "STEP-01-RECV-USER-QUERY"
STEP_AGENT_PIPELINE = "STEP-02-AGENT-PIPELINE"
STEP_LLM_FALLBACK = "STEP-03-LLM-FALLBACK"
STEP_LLM_NLU = "STEP-02A-LLM-NLU"
STEP_LLM_SEARCH_RERANK = "STEP-02B-LLM-SEARCH-RERANK"
STEP_LLM_RESPOND = "STEP-03-LLM-RESPOND"
STEP_FINAL_RESPONSE = "STEP-04-FINAL-RESPONSE"
STEP_HTTP = "STEP-00-HTTP"
STARTUP_LANDMARK_PRELOAD_SESSION_ID = "startup_landmarks_preload"

LLM_TIMEOUT = httpx.Timeout(90.0)
_LLM_TIMEOUT_RETRY_ATTEMPTS = 2
_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_TOOL_ARGUMENT_KEY_ALIASES: dict[str, dict[str, str]] = {
    "get_house_by_id": {"id": "house_id"},
    "get_house_listings": {"id": "house_id"},
    "rent_house": {"id": "house_id"},
    "terminate_rental": {"id": "house_id"},
    "take_offline": {"id": "house_id"},
}
_SOFT_BOOL_DEFAULT_FALSE_KEYS = {"prefer_spacious", "prioritize_subway_distance", "prioritize_commute", "value_for_money"}
_LLM_CONTEXT_HARD_KEYS = ("district", "layout", "budget_min", "budget_max", "max_subway_dist", "rent_type", "area")
_LLM_CONTEXT_SOFT_KEYS = ("decoration", "amenities", "preferred_tags", "avoid_tags")
_LLM_CONTEXT_TAG_LIMIT = 80
_LLM_CONTEXT_HOUSE_LIMIT = 10
_LLM_CONTEXT_TAG_NEED_LIMIT = 10
_SEARCH_RERANK_HOUSE_CONTEXT_LIMIT = 10
_SEARCH_RERANK_TAG_LIMIT = 8
_DEFAULT_SEARCH_RERANK_HOUSE_FIELDS = (
    "house_id",
    "community",
    "district",
    "rent",
    "price",
    "layout",
    "bedrooms",
    "livingrooms",
    "bathrooms",
    "area",
    "area_sqm",
    "subway",
    "subway_station",
    "subway_distance",
    "commute_to_xierqi_min",
    "commute_to_xierqi",
    "status",
    "listing_platform",
    "decoration",
    "elevator",
    "orientation",
    "available_date",
    "available_from",
    "rental_type",
    "utilities_type",
    "tags",
    "nearby_landmarks",
)
_DEFAULT_SEARCH_RERANK_LANDMARK_FIELDS = (
    "id",
    "name",
    "category",
    "district",
    "type",
    "type_name",
    "distance",
    "nearby_subway",
)
_HOUSE_CONTEXT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "price": ("price", "rent"),
    "rent": ("rent", "price"),
    "area_sqm": ("area_sqm", "area"),
    "area": ("area", "area_sqm"),
    "commute_to_xierqi": ("commute_to_xierqi", "commute_to_xierqi_min"),
    "commute_to_xierqi_min": ("commute_to_xierqi_min", "commute_to_xierqi"),
    "available_from": ("available_from", "available_date"),
    "available_date": ("available_date", "available_from"),
    "subway": ("subway", "nearest_subway"),
    "subway_station": ("subway_station", "nearest_subway"),
}
_SEARCH_RERANK_HOUSE_FIELDS: tuple[str, ...] = _DEFAULT_SEARCH_RERANK_HOUSE_FIELDS
_SEARCH_RERANK_LANDMARK_FIELDS: tuple[str, ...] = _DEFAULT_SEARCH_RERANK_LANDMARK_FIELDS
_SEARCH_RERANK_LANDMARK_ITEM_LIMIT = 2
_TOOL_OPTIONAL_PARAM_PRIORITY = (
    "house_id",
    "listing_platform",
    "district",
    "area",
    "community",
    "landmark_id",
    "q",
    "name",
    "id",
    "min_price",
    "max_price",
    "bedrooms",
    "rental_type",
    "decoration",
    "max_subway_dist",
    "min_area",
    "commute_to_xierqi_max",
    "subway_station",
    "utilities_type",
    "page",
    "page_size",
    "max_distance",
    "max_distance_m",
    "type",
)
_TOOL_OPTIONAL_PARAM_LIMIT = 10
_HOUSE_ID_EXTRACT_PATTERN = re.compile(r"[A-Z]{2,4}_?\d{1,8}", re.IGNORECASE)
_LLM_NLU_PARAM_KEY_ALIASES = {
    "district": "district",
    "d": "district",
    "bedrooms": "bedrooms",
    "b": "bedrooms",
    "min_price": "min_price",
    "min": "min_price",
    "max_price": "max_price",
    "max": "max_price",
    "decoration": "decoration",
    "dec": "decoration",
    "max_subway_dist": "max_subway_dist",
    "sub": "max_subway_dist",
    "rental_type": "rental_type",
    "rt": "rental_type",
    "min_area": "min_area",
    "area": "min_area",
    "elevator": "elevator",
    "el": "elevator",
}
_LLM_NLU_TAG_KEY_ALIASES = {
    "must": "must",
    "m": "must",
    "avoid": "avoid",
    "a": "avoid",
    "prefer": "prefer",
    "p": "prefer",
}

_LLM_COMPACT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "landmark",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["list", "search", "by_id", "by_name", "stats"]},
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "q": {"type": "string"},
                    "district": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "house_query",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "get",
                            "listings",
                            "by_community",
                            "by_platform",
                            "nearby_landmarks",
                            "nearby_houses",
                            "stats",
                        ],
                    },
                    "house_id": {"type": "string"},
                    "landmark_id": {"type": "string"},
                    "community": {"type": "string"},
                    "listing_platform": {"type": "string"},
                    "district": {"type": "string"},
                    "area": {"type": "string"},
                    "page": {"type": "integer"},
                    "page_size": {"type": "integer"},
                    "min_price": {"type": "integer"},
                    "max_price": {"type": "integer"},
                    "bedrooms": {"type": "string"},
                    "rental_type": {"type": "string"},
                    "decoration": {"type": "string"},
                    "max_subway_dist": {"type": "integer"},
                    "min_area": {"type": "integer"},
                    "max_distance": {"type": "number"},
                    "max_distance_m": {"type": "number"},
                    "type": {"type": "string"},
                },
                "required": ["op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "house_action",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["rent", "terminate", "offline"]},
                    "house_id": {"type": "string"},
                    "listing_platform": {"type": "string"},
                },
                "required": ["action", "house_id", "listing_platform"],
            },
        },
    },
]


def _parse_http_body_for_io(raw_body: bytes, content_type: str | None) -> Any:
    if not raw_body:
        return "<empty>"

    text = raw_body.decode("utf-8", errors="replace")
    if content_type and "application/json" in content_type.lower():
        parsed_obj = _extract_json_object(text)
        if parsed_obj is not None:
            return parsed_obj
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass
    return text


def _resolve_http_session_id(request: Request, raw_body: bytes, content_type: str | None) -> str:
    header_session = request.headers.get("Session-ID") or request.headers.get("session-id")
    if isinstance(header_session, str) and header_session.strip():
        return header_session.strip()

    query_session = request.query_params.get("session_id")
    if isinstance(query_session, str) and query_session.strip():
        return query_session.strip()

    if content_type and "application/json" in content_type.lower() and raw_body:
        parsed = _extract_json_object(raw_body.decode("utf-8", errors="replace"))
        candidate = parsed.get("session_id") if isinstance(parsed, dict) else None
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    context_session = get_log_context().get("session_id", "-")
    if isinstance(context_session, str) and context_session.strip():
        return context_session.strip()
    return "-"


def _is_debug_agent_io_path(path: str) -> bool:
    return path.startswith("/debug/agent-io")

def _resolve_agent_http_io_log_path() -> Path:
    env_path = os.getenv("AGENT_HTTP_IO_LOG_PATH")
    if isinstance(env_path, str) and env_path.strip():
        configured_path = Path(env_path.strip())
        if configured_path.is_absolute():
            return configured_path
        cwd_path = Path.cwd() / configured_path
        project_path = Path(__file__).resolve().parents[1] / configured_path
        if cwd_path.exists():
            return cwd_path
        if project_path.exists():
            return project_path
        return cwd_path

    logger = logging.getLogger("agent.http.io")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            filename = getattr(handler, "baseFilename", "")
            if isinstance(filename, str) and filename:
                return Path(filename)

    configured_path = Path("agent_http_io.log")
    if configured_path.is_absolute():
        return configured_path

    cwd_path = Path.cwd() / configured_path
    project_path = Path(__file__).resolve().parents[1] / configured_path
    if cwd_path.exists():
        return cwd_path
    if project_path.exists():
        return project_path
    return cwd_path

def _read_agent_http_io_entries(limit: int = 200) -> list[dict[str, Any]]:
    path = _resolve_agent_http_io_log_path()
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        brace_idx = line.find("{")
        if brace_idx < 0:
            continue
        raw_json = line[brace_idx:]
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _stage_name(entry: dict[str, Any]) -> str:
    event = str(entry.get("event", ""))
    path = str(entry.get("path", ""))
    method = str(entry.get("method", ""))

    if event == "http.agent_io.request" and method == "POST" and path == "/api/v1/chat":
        return "agent接收用户输入"
    if event == "http.agent_io.response" and method == "POST" and path == "/api/v1/chat":
        return "agent最终返回用户"
    if event == "http.agent_io" and method == "POST" and path == "/api/v1/chat":
        return "agent接收用户输入 + agent最终返回用户"
    if event == "http.agent_io.llm.request":
        return "agent发往LLM的输入"
    if event == "http.agent_io.llm.response":
        return "LLM返回的输出"
    if event == "http.agent_io.api.request":
        return "agent调用API"
    if event == "http.agent_io.api.response":
        return "agent得到API输出"
    if event.endswith(".error"):
        return "调用错误"
    return "其他"


def _looks_like_json_container_text(value: str) -> bool:
    text = value.strip()
    if not text or text.endswith("...(truncated)"):
        return False
    return (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))


def _normalize_agent_io_value_for_display(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return value

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                output[key] = _normalize_agent_io_value_for_display(item, depth=depth + 1)
        return output

    if isinstance(value, list):
        return [_normalize_agent_io_value_for_display(item, depth=depth + 1) for item in value]

    if isinstance(value, str) and _looks_like_json_container_text(value):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return _normalize_agent_io_value_for_display(parsed, depth=depth + 1)

    return value


def _normalize_agent_io_entry_for_display(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_agent_io_value_for_display(entry, depth=0)
    return normalized if isinstance(normalized, dict) else entry


def _compact_agent_io_entry_for_ui(entry: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "stage",
        "event",
        "session_id",
        "trace_id",
        "method",
        "path",
        "url",
        "status_code",
        "llm_stage",
        "duration_ms",
        "error",
        "error_type",
    ):
        value = entry.get(key)
        if value is not None and value != "":
            compact[key] = value

    if "request_body" in entry:
        compact["request_body"] = entry.get("request_body")
    if "response_body" in entry:
        compact["response_body"] = entry.get("response_body")
    if "body" in entry:
        compact["body"] = entry.get("body")

    return compact


def _format_ops_index_for_prompt(ops: list[dict[str, Any]], *, max_ops: int = 30) -> str:
    if not ops:
        return "-"
    lines: list[str] = []
    for row in ops[:max_ops]:
        if not isinstance(row, dict):
            continue
        op_id = row.get("operationId")
        args = row.get("args")
        if not isinstance(op_id, str) or not op_id:
            continue
        if isinstance(args, list):
            arg_names = [str(item).strip() for item in args if isinstance(item, str) and item.strip()]
        else:
            arg_names = []
        lines.append(f"- {op_id}({','.join(arg_names)})")
    return "\n".join(lines) if lines else "-"


def _flatten_prompt_context_lines(value: Any, *, prefix: str = "", lines: list[str], max_lines: int = 80) -> None:
    if len(lines) >= max_lines:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if len(lines) >= max_lines:
                return
            if not isinstance(key, str):
                continue
            next_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_prompt_context_lines(item, prefix=next_prefix, lines=lines, max_lines=max_lines)
        return
    if isinstance(value, list):
        if not value:
            return
        scalar_items = [item for item in value if isinstance(item, (str, int, float, bool))]
        if scalar_items and len(scalar_items) == len(value):
            joined = ",".join(str(item) for item in scalar_items[:12])
            lines.append(f"{prefix}={joined}")
            return
        lines.append(f"{prefix}=[{len(value)} items]")
        return
    if value is None:
        return
    lines.append(f"{prefix}={value}")


def _format_context_facts_for_prompt(context_facts: dict[str, Any] | None) -> str:
    if not isinstance(context_facts, dict) or not context_facts:
        return "-"
    lines: list[str] = []
    _flatten_prompt_context_lines(context_facts, lines=lines, max_lines=80)
    return "\n".join(lines) if lines else "-"


def _format_tool_results_for_prompt(
    tool_results: list[dict[str, Any]] | None,
    *,
    max_tools: int = 8,
    max_lines: int = 120,
) -> str:
    if not isinstance(tool_results, list) or not tool_results:
        return "-"

    lines: list[str] = []
    for idx, row in enumerate(tool_results[:max_tools], start=1):
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        success = row.get("success")
        method = row.get("method")
        status_code = row.get("status_code")
        url = row.get("url")
        lines.append(
            (
                f"[{idx}] name={name if isinstance(name, str) and name else '-'} "
                f"success={success} method={method if isinstance(method, str) and method else '-'} "
                f"status={status_code if status_code is not None else '-'} "
                f"url={url if isinstance(url, str) and url else '-'}"
            )
        )

        output_lines: list[str] = []
        _flatten_prompt_context_lines(row.get("output"), prefix=f"[{idx}].output", lines=output_lines, max_lines=24)
        if output_lines:
            lines.extend(output_lines)

        if len(lines) >= max_lines:
            lines = lines[:max_lines]
            lines.append("...(more tool result lines omitted)")
            break

    return "\n".join(lines) if lines else "-"


def _parse_csv_tokens(raw: Any, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw, str):
        return fallback
    parts = [token.strip() for token in raw.split(",")]
    output: list[str] = []
    seen: set[str] = set()
    for token in parts:
        if not token or token in seen:
            continue
        seen.add(token)
        output.append(token)
    return tuple(output) if output else fallback


def _clamp_positive_int(value: Any, *, fallback: int, min_value: int = 1, max_value: int = 10) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(min_value, min(max_value, parsed))


def _configure_search_rerank_context_settings(cfg: AgentSettings) -> None:
    global _SEARCH_RERANK_HOUSE_FIELDS
    global _SEARCH_RERANK_LANDMARK_FIELDS
    global _SEARCH_RERANK_LANDMARK_ITEM_LIMIT

    _SEARCH_RERANK_HOUSE_FIELDS = _parse_csv_tokens(
        getattr(cfg, "rerank_house_context_fields", ""),
        fallback=_DEFAULT_SEARCH_RERANK_HOUSE_FIELDS,
    )
    _SEARCH_RERANK_LANDMARK_FIELDS = _parse_csv_tokens(
        getattr(cfg, "rerank_landmark_context_fields", ""),
        fallback=_DEFAULT_SEARCH_RERANK_LANDMARK_FIELDS,
    )
    _SEARCH_RERANK_LANDMARK_ITEM_LIMIT = _clamp_positive_int(
        getattr(cfg, "rerank_landmark_item_limit", 2),
        fallback=2,
        min_value=1,
        max_value=10,
    )


def _build_model_base_url(model_ip: str) -> str:
    if model_ip.startswith(("http://", "https://")):
        return model_ip
    return f"http://{model_ip}:8888"


def _normalize_error_payload(exc: Exception) -> dict[str, str]:
    message = str(exc).strip()
    if not message:
        message = repr(exc)
    timeout_reason: str | None = None
    if isinstance(exc, httpx.ReadTimeout):
        timeout_reason = f"LLM read timeout after {LLM_TIMEOUT.read}s while waiting for response data"
    elif isinstance(exc, httpx.ConnectTimeout):
        timeout_reason = f"LLM connect timeout after {LLM_TIMEOUT.connect}s while establishing TCP connection"
    elif isinstance(exc, httpx.WriteTimeout):
        timeout_reason = f"LLM write timeout after {LLM_TIMEOUT.write}s while sending request payload"
    elif isinstance(exc, httpx.PoolTimeout):
        timeout_reason = f"LLM pool timeout after {LLM_TIMEOUT.pool}s while acquiring HTTP connection"

    if timeout_reason:
        message = f"{timeout_reason}; raw_error={message}"
    return {
        "error": message,
        "error_type": type(exc).__name__,
    }


async def _forward_chat_completion(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    step: str = STEP_LLM_FALLBACK,
    llm_stage: str = "generic",
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if session_id:
        headers["Session-ID"] = session_id

    user_messages = _normalize_messages_for_eval(messages)
    payload = {
        "model": "",
        "messages": user_messages,
        "tools": tools if tools is not None else [],
        "stream": False,
    }
    target_url = f"{_build_model_base_url(model_ip)}/v1/chat/completions"
    started = time.perf_counter()
    log_event(
        LOGGER,
        "chat.llm.forward.request",
        step=step,
        llm_stage=llm_stage,
        model_ip=model_ip,
        payload_preview=preview_payload(payload),
    )
    log_json_event(
        HTTP_IO_LOGGER,
        {
            **get_log_context(),
            "event": "http.agent_io.llm.request",
            "llm_stage": llm_stage,
            "method": "POST",
            "url": target_url,
            "session_id": session_id or "-",
            "headers": headers,
            "request_content_type": "application/json",
            "request_body": payload,
        },
    )

    data: dict[str, Any]
    response_body: Any = {}
    resp: httpx.Response | None = None
    try:
        for attempt in range(1, _LLM_TIMEOUT_RETRY_ATTEMPTS + 1):
            try:
                resp = await _llm_post(
                    http_client,
                    url=target_url,
                    payload=payload,
                    headers=headers,
                )
                response_body = _parse_llm_http_body(resp)
                log_json_event(
                    HTTP_IO_LOGGER,
                    {
                        **get_log_context(),
                        "event": "http.agent_io.llm.response",
                        "llm_stage": llm_stage,
                        "method": "POST",
                        "url": target_url,
                        "session_id": session_id or "-",
                        "status_code": resp.status_code,
                        "response_content_type": getattr(resp, "headers", {}).get("content-type", "application/json"),
                        "response_body": response_body,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                        "attempt": attempt,
                    },
                )
                resp.raise_for_status()
                if isinstance(response_body, dict):
                    data = response_body
                else:
                    parsed = _extract_json_object(str(response_body))
                    if not parsed:
                        raise ValueError("LLM response body is not a valid JSON object")
                    data = parsed
                break
            except httpx.TimeoutException as exc:
                if attempt >= _LLM_TIMEOUT_RETRY_ATTEMPTS:
                    raise
                log_event(
                    LOGGER,
                    "chat.llm.forward.retry",
                    step=step,
                    llm_stage=llm_stage,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    reason=type(exc).__name__,
                )
                log_json_event(
                    HTTP_IO_LOGGER,
                    {
                        **get_log_context(),
                        "event": "http.agent_io.llm.retry",
                        "llm_stage": llm_stage,
                        "method": "POST",
                        "url": target_url,
                        "session_id": session_id or "-",
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        **_normalize_error_payload(exc),
                    },
                )
    except Exception as exc:
        log_json_event(
            HTTP_IO_LOGGER,
            {
                **get_log_context(),
                "event": "http.agent_io.llm.error",
                "llm_stage": llm_stage,
                "method": "POST",
                "url": target_url,
                "session_id": session_id or "-",
                **_normalize_error_payload(exc),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        raise
    log_event(
        LOGGER,
        "chat.llm.forward.response",
        step=step,
        llm_stage=llm_stage,
        status_code=resp.status_code if resp is not None else 0,
        body_preview=preview_payload(data),
    )

    return data


def _parse_llm_http_body(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        text_body = resp.text.strip() if isinstance(resp.text, str) else ""
        return {"raw": text_body} if text_body else {}


async def _llm_post(
    http_client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    try:
        return await http_client.post(
            url,
            json=payload,
            headers=headers,
            timeout=LLM_TIMEOUT,
        )
    except TypeError as exc:
        # Some unit-test stubs do not accept timeout kwargs.
        if "timeout" not in str(exc):
            raise
        return await http_client.post(
            url,
            json=payload,
            headers=headers,
        )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    candidates = [cleaned]
    if "```" in cleaned:
        candidates.append(cleaned.replace("```json", "").replace("```", "").strip())

    left = cleaned.find("{")
    right = cleaned.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidates.append(cleaned[left : right + 1])

    for item in candidates:
        try:
            value = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


@lru_cache(maxsize=1)
def _load_compact_tool_schema_payload() -> dict[str, Any]:
    compact_schema_path = Path(__file__).resolve().parents[1] / "llm_tool_schema.json"
    try:
        compact_payload = json.loads(compact_schema_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"llm_tool_schema.json not found: {compact_schema_path}") from exc
    except ValueError as exc:
        raise RuntimeError(f"llm_tool_schema.json is invalid JSON: {compact_schema_path}") from exc
    if not isinstance(compact_payload, dict):
        raise RuntimeError(f"llm_tool_schema.json must be a JSON object: {compact_schema_path}")
    return compact_payload


@lru_cache(maxsize=1)
def _load_tool_schema_summary() -> str:
    compact_payload = _load_compact_tool_schema_payload()
    compact_summary = _summarize_compact_tool_schema(compact_payload)
    if not compact_summary:
        compact_schema_path = Path(__file__).resolve().parents[1] / "llm_tool_schema.json"
        raise RuntimeError(f"llm_tool_schema.json has no valid operations: {compact_schema_path}")
    return compact_summary


@lru_cache(maxsize=1)
def _load_tool_argument_specs() -> dict[str, dict[str, dict[str, Any]]]:
    payload = _load_compact_tool_schema_payload()
    operations = payload.get("operations")
    if not isinstance(operations, list):
        return {}

    spec_map: dict[str, dict[str, dict[str, Any]]] = {}
    for item in operations:
        if not isinstance(item, dict):
            continue
        operation_id = item.get("operationId")
        if not isinstance(operation_id, str) or not operation_id:
            continue
        param_defs = _collect_param_defs(item)
        if not param_defs:
            continue
        op_specs: dict[str, dict[str, Any]] = {}
        for spec in param_defs:
            name = spec.get("name")
            if isinstance(name, str) and name:
                op_specs[name] = spec
        if op_specs:
            spec_map[operation_id] = op_specs
    return spec_map


def _summarize_compact_tool_schema(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    operations = payload.get("operations")
    if not isinstance(operations, list):
        return ""

    lines: list[str] = []
    for item in operations:
        if not isinstance(item, dict):
            continue
        operation_id = item.get("operationId")
        if not isinstance(operation_id, str) or not operation_id:
            continue

        method = item.get("method")
        path = item.get("path")
        intent = item.get("intent")
        param_defs = _collect_param_defs(item)
        output_tags = item.get("output_tags")
        hints = item.get("hints")

        required = [spec["name"] for spec in param_defs if spec.get("required") is True]
        params = [spec["name"] for spec in param_defs]
        method_text = method.upper() if isinstance(method, str) and method else "GET"
        path_text = path if isinstance(path, str) and path else "-"
        required_text = ",".join(required) if required else "-"
        params_text = ",".join(params) if params else "-"
        argdefs_text = _format_param_defs_for_prompt(param_defs)
        outputs_text = ",".join(output_tags) if isinstance(output_tags, list) and output_tags else "-"
        intent_text = intent if isinstance(intent, str) and intent else "search"
        hint_text = hints if isinstance(hints, str) and hints else "-"
        lines.append(
            f"{operation_id}|{method_text} {path_text}|intent={intent_text}|required={required_text}|params={params_text}|argdefs={argdefs_text}|outputs={outputs_text}|hints={hint_text}"
        )

    enums = payload.get("enums")
    if isinstance(enums, dict):
        enum_chunks: list[str] = []
        for key, value in enums.items():
            if isinstance(value, list) and value:
                enum_chunks.append(f"{key}={','.join(str(v) for v in value)}")
        if enum_chunks:
            lines.append("enums|" + ";".join(enum_chunks))

    return "\n".join(lines[:40])


def _collect_param_defs(item: dict[str, Any]) -> list[dict[str, Any]]:
    raw_defs = item.get("param_defs")
    normalized_defs: list[dict[str, Any]] = []
    if isinstance(raw_defs, list):
        for raw in raw_defs:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                continue
            param_type = raw.get("type")
            if not isinstance(param_type, str) or not param_type:
                param_type = "string"
            location = raw.get("in")
            if not isinstance(location, str) or not location:
                location = "query"
            required = raw.get("required") is True
            enum_values = raw.get("enum")
            if not isinstance(enum_values, list):
                enum_values = None
            description = raw.get("description")
            if not isinstance(description, str):
                description = ""
            normalized_defs.append(
                {
                    "name": name,
                    "type": param_type,
                    "in": location,
                    "required": required,
                    "enum": enum_values,
                    "description": description.strip(),
                }
            )
    if normalized_defs:
        return normalized_defs

    # Backward compatibility: synthesize from params + required if param_defs are absent.
    raw_params = item.get("params")
    raw_required = item.get("required")
    params = [name for name in raw_params if isinstance(name, str) and name] if isinstance(raw_params, list) else []
    required_set = set(name for name in raw_required if isinstance(name, str) and name) if isinstance(raw_required, list) else set()
    return [
        {
            "name": name,
            "type": "string",
            "in": "query",
            "required": name in required_set,
            "enum": None,
            "description": "",
        }
        for name in params
    ]


@lru_cache(maxsize=1)
def _load_llm_tools() -> list[dict[str, Any]]:
    # Keep a fixed compact tool schema to control prompt token usage.
    return _LLM_COMPACT_TOOLS


def _compact_tools_for_llm(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in tools:
        function_node = item.get("function")
        if not isinstance(function_node, dict):
            continue
        name = function_node.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = _compact_tool_parameters(function_node.get("parameters"))
        description = function_node.get("description")
        if not isinstance(description, str) or not description.strip():
            compact_description = name
        else:
            # Keep only the first sentence/segment to reduce token usage while
            # preserving endpoint hints (e.g. "POST /api/...").
            compact_description = description.strip().split("。", 1)[0].split("|", 1)[0].strip()
            if len(compact_description) > 96:
                compact_description = compact_description[:96]
        compacted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": compact_description,
                    "parameters": parameters,
                },
            }
        )
    return compacted


def _compact_tool_parameters(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    properties = raw.get("properties")
    compact_props: dict[str, Any] = {}
    if isinstance(properties, dict):
        required_raw = raw.get("required")
        required_set = (
            {item for item in required_raw if isinstance(item, str) and item in properties}
            if isinstance(required_raw, list)
            else set()
        )
        selected_keys: list[str] = []
        seen_keys: set[str] = set()

        for key in properties:
            if not isinstance(key, str) or not key or key not in required_set or key in seen_keys:
                continue
            seen_keys.add(key)
            selected_keys.append(key)

        optional_count = 0
        for key in _TOOL_OPTIONAL_PARAM_PRIORITY:
            if key in seen_keys or key not in properties:
                continue
            seen_keys.add(key)
            selected_keys.append(key)
            optional_count += 1
            if optional_count >= _TOOL_OPTIONAL_PARAM_LIMIT:
                break

        if optional_count < _TOOL_OPTIONAL_PARAM_LIMIT:
            for key in properties:
                if not isinstance(key, str) or not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                selected_keys.append(key)
                optional_count += 1
                if optional_count >= _TOOL_OPTIONAL_PARAM_LIMIT:
                    break

        for key in selected_keys:
            spec = properties.get(key)
            if not isinstance(spec, dict):
                compact_props[key] = {"type": "string"}
                continue
            entry: dict[str, Any] = {}
            param_type = spec.get("type")
            if isinstance(param_type, str) and param_type:
                entry["type"] = param_type
            else:
                entry["type"] = "string"
            enum_values = spec.get("enum")
            if isinstance(enum_values, list) and enum_values and (key in required_set or len(enum_values) <= 5):
                entry["enum"] = [v for v in enum_values if isinstance(v, (str, int, float, bool))][:5]
            compact_props[key] = entry

    compact: dict[str, Any] = {
        "type": "object",
        "properties": compact_props,
        "additionalProperties": False,
    }
    required = raw.get("required")
    if isinstance(required, list) and required:
        compact["required"] = [x for x in required if isinstance(x, str) and x in compact_props]
    return compact


@lru_cache(maxsize=1)
def _load_operation_intents() -> dict[str, str]:
    payload = _load_compact_tool_schema_payload()
    operations = payload.get("operations")
    if not isinstance(operations, list):
        return {}

    mapping: dict[str, str] = {}
    for item in operations:
        if not isinstance(item, dict):
            continue
        operation_id = item.get("operationId")
        intent = item.get("intent")
        if isinstance(operation_id, str) and operation_id and isinstance(intent, str) and intent:
            mapping[operation_id] = intent
    return mapping


def _format_param_defs_for_prompt(param_defs: list[dict[str, Any]]) -> str:
    if not param_defs:
        return "-"

    chunks: list[str] = []
    for spec in param_defs:
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            continue
        param_type = spec.get("type")
        if not isinstance(param_type, str) or not param_type:
            param_type = "string"
        required_suffix = "!" if spec.get("required") is True else ""
        location = spec.get("in")
        location_text = f"@{location}" if isinstance(location, str) and location else ""

        enum_text = ""
        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and enum_values:
            enum_joined = "/".join(str(v) for v in enum_values[:4])
            if len(enum_values) > 4:
                enum_joined += "/..."
            enum_text = f"={enum_joined}"

        description_text = ""
        description = spec.get("description")
        if spec.get("required") is True and isinstance(description, str) and description:
            cleaned = description.replace("\n", " ").replace("|", "/").strip()
            if len(cleaned) > 14:
                cleaned = cleaned[:14] + "..."
            description_text = f"({cleaned})"

        chunks.append(f"{name}:{param_type}{required_suffix}{location_text}{enum_text}{description_text}")
    return ";".join(chunks) if chunks else "-"


@lru_cache(maxsize=1)
def _load_tool_required_params() -> dict[str, set[str]]:
    payload = _load_compact_tool_schema_payload()
    operations = payload.get("operations")
    if not isinstance(operations, list):
        return {}

    mapping: dict[str, set[str]] = {}
    for item in operations:
        if not isinstance(item, dict):
            continue
        operation_id = item.get("operationId")
        if not isinstance(operation_id, str) or not operation_id:
            continue
        param_defs = _collect_param_defs(item)
        required = {spec["name"] for spec in param_defs if spec.get("required") is True and isinstance(spec.get("name"), str)}
        mapping[operation_id] = required
    return mapping


@lru_cache(maxsize=1)
def _load_llm_ops_prompt_index() -> list[dict[str, Any]]:
    payload = _load_compact_tool_schema_payload()
    operations = payload.get("operations")
    if not isinstance(operations, list):
        return []

    ops: list[dict[str, Any]] = []
    for item in operations:
        if not isinstance(item, dict):
            continue
        operation_id = item.get("operationId")
        if not isinstance(operation_id, str) or not operation_id:
            continue
        param_defs = _collect_param_defs(item)
        required = [spec["name"] for spec in param_defs if spec.get("required") is True and isinstance(spec.get("name"), str)]
        all_param_names = [spec["name"] for spec in param_defs if isinstance(spec.get("name"), str)]

        args: list[str] = []
        seen: set[str] = set()
        for name in required:
            if name in seen:
                continue
            seen.add(name)
            args.append(name)
        for name in _TOOL_OPTIONAL_PARAM_PRIORITY:
            if name in seen or name not in all_param_names:
                continue
            seen.add(name)
            args.append(name)
            if len(args) >= 12:
                break
        if len(args) < 12:
            for name in all_param_names:
                if name in seen:
                    continue
                seen.add(name)
                args.append(name)
                if len(args) >= 12:
                    break

        ops.append({"operationId": operation_id, "args": args})
    return ops


def _prune_empty_for_llm(value: Any, *, drop_false_keys: set[str] | None = None, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, raw in value.items():
            if not isinstance(key, str):
                continue
            pruned = _prune_empty_for_llm(raw, drop_false_keys=drop_false_keys, parent_key=key)
            if pruned is None:
                continue
            if isinstance(pruned, bool) and pruned is False and drop_false_keys and key in drop_false_keys:
                continue
            if pruned == "":
                continue
            if isinstance(pruned, (list, dict)) and not pruned:
                continue
            output[key] = pruned
        return output or None

    if isinstance(value, list):
        output_list: list[Any] = []
        for item in value:
            pruned = _prune_empty_for_llm(item, drop_false_keys=drop_false_keys, parent_key=parent_key)
            if pruned is None:
                continue
            if pruned == "":
                continue
            if isinstance(pruned, (list, dict)) and not pruned:
                continue
            output_list.append(pruned)
        return output_list or None

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _compact_tag_need_payload(value: Any) -> dict[str, list[str]]:
    payload = value if isinstance(value, dict) else {}
    result: dict[str, list[str]] = {"must": [], "avoid": [], "prefer": []}
    for key in ("must", "avoid", "prefer"):
        raw = payload.get(key)
        if not isinstance(raw, list):
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            cleaned.append(token)
            if len(cleaned) >= _LLM_CONTEXT_TAG_NEED_LIMIT:
                break
        result[key] = cleaned
    return result


def _value_from_item(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _round_float_for_context(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    return value


def _normalize_house_context_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _round_float_for_context(value)
    if isinstance(value, list):
        if field == "tags":
            return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()][: _SEARCH_RERANK_TAG_LIMIT]
        if field in {"pros", "cons"}:
            return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()][:5]
        cleaned: list[Any] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (bool, int, str)):
                cleaned.append(item)
            elif isinstance(item, float):
                cleaned.append(_round_float_for_context(item))
            elif isinstance(item, dict) and item:
                cleaned.append(item)
            if len(cleaned) >= 6:
                break
        return cleaned or None
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or item is None:
                continue
            if isinstance(item, (bool, int, str)):
                compact[key] = item
            elif isinstance(item, float):
                compact[key] = _round_float_for_context(item)
            if len(compact) >= 8:
                break
        return compact or None
    return str(value)


def _house_context_field_value(item: Any, key: str) -> Any:
    aliases = _HOUSE_CONTEXT_FIELD_ALIASES.get(key, (key,))
    for alias in aliases:
        value = _value_from_item(item, alias)
        if value is not None:
            return value
    return None


def _landmark_context_field_value(
    *,
    field: str,
    item: dict[str, Any],
    landmark: dict[str, Any],
    details: dict[str, Any],
    output: dict[str, Any],
) -> Any:
    if field == "distance":
        return item.get("distance")
    if field == "type":
        return output.get("type") or details.get("type")
    if field == "type_name":
        return details.get("type_name")
    if field == "nearby_subway":
        return details.get("nearby_subway")
    if field == "landmark_id":
        return details.get("landmark_id") or landmark.get("id")

    if field in item and item.get(field) is not None:
        return item.get(field)
    if field in landmark and landmark.get(field) is not None:
        return landmark.get(field)
    if field in details and details.get(field) is not None:
        return details.get(field)
    if field in output and output.get(field) is not None:
        return output.get(field)
    return None


def _collect_nearby_landmarks_context(tool_results: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(tool_results, list) or not tool_results:
        return {}

    fields = _SEARCH_RERANK_LANDMARK_FIELDS or _DEFAULT_SEARCH_RERANK_LANDMARK_FIELDS
    item_limit = max(1, _SEARCH_RERANK_LANDMARK_ITEM_LIMIT)
    by_community: dict[str, list[dict[str, Any]]] = {}
    seen_keys: dict[str, set[str]] = {}

    for row in tool_results:
        if not isinstance(row, dict) or row.get("success") is not True:
            continue
        name = str(row.get("name", ""))
        url = str(row.get("url", ""))
        if "nearby_landmarks" not in name and "/nearby_landmarks" not in url:
            continue

        output = row.get("output")
        if not isinstance(output, dict):
            continue
        community = output.get("community")
        if not isinstance(community, str) or not community.strip():
            continue
        items = output.get("items")
        if not isinstance(items, list) or not items:
            continue

        bucket = by_community.setdefault(community, [])
        seen = seen_keys.setdefault(community, set())
        for raw in items:
            if not isinstance(raw, dict):
                continue
            landmark = raw.get("landmark")
            if not isinstance(landmark, dict):
                continue
            details = landmark.get("details")
            if not isinstance(details, dict):
                details = {}

            compact: dict[str, Any] = {}
            for field in fields:
                value = _landmark_context_field_value(
                    field=field,
                    item=raw,
                    landmark=landmark,
                    details=details,
                    output=output,
                )
                normalized = _normalize_house_context_value(field, value)
                if normalized is not None:
                    compact[field] = normalized

            if not compact:
                continue
            marker = str(compact.get("id") or compact.get("name") or compact)
            if marker in seen:
                continue
            seen.add(marker)
            bucket.append(compact)
            if len(bucket) >= item_limit:
                break

    return by_community


def _house_context_row(item: Any, *, nearby_landmarks_by_community: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    house_id = _value_from_item(item, "house_id")
    if not isinstance(house_id, str) or not house_id:
        return {}

    row: dict[str, Any] = {"house_id": house_id}

    fields = _SEARCH_RERANK_HOUSE_FIELDS or _DEFAULT_SEARCH_RERANK_HOUSE_FIELDS
    for key in fields:
        if key == "house_id":
            continue
        if key == "nearby_landmarks":
            if isinstance(nearby_landmarks_by_community, dict) and nearby_landmarks_by_community:
                community = _house_context_field_value(item, "community")
                if isinstance(community, str) and community:
                    landmark_rows = nearby_landmarks_by_community.get(community)
                    if isinstance(landmark_rows, list) and landmark_rows:
                        row["nearby_landmarks"] = landmark_rows[: max(1, _SEARCH_RERANK_LANDMARK_ITEM_LIMIT)]
            continue

        value = _house_context_field_value(item, key)
        normalized = _normalize_house_context_value(key, value)
        if normalized is not None:
            row[key] = normalized

    return row


def _house_context_row_to_lite(row: dict[str, Any]) -> HouseLite | None:
    if not isinstance(row, dict):
        return None
    house_id = row.get("house_id")
    if not isinstance(house_id, str) or not house_id:
        return None

    payload: dict[str, Any] = {"house_id": house_id}
    if row.get("rent") is not None:
        payload["rent"] = row.get("rent")
    elif row.get("price") is not None:
        payload["rent"] = row.get("price")
    if row.get("area") is not None:
        payload["area"] = row.get("area")
    elif row.get("area_sqm") is not None:
        payload["area"] = row.get("area_sqm")
    if row.get("commute_to_xierqi_min") is not None:
        payload["commute_to_xierqi_min"] = row.get("commute_to_xierqi_min")
    elif row.get("commute_to_xierqi") is not None:
        payload["commute_to_xierqi_min"] = row.get("commute_to_xierqi")
    if row.get("available_date") is not None:
        payload["available_date"] = row.get("available_date")
    elif row.get("available_from") is not None:
        payload["available_date"] = row.get("available_from")

    for key in (
        "layout",
        "business_area",
        "district",
        "community",
        "subway_distance",
        "status",
        "tags",
        "decoration",
        "elevator",
        "orientation",
        "listing_platform",
        "distance_to_landmark",
        "walking_distance",
        "walking_duration",
    ):
        value = row.get(key)
        if value is None:
            continue
        payload[key] = value

    try:
        return HouseLite.model_validate(payload)
    except Exception:
        return None


def _set_state_house_context_top10(state: Any, rows: list[dict[str, Any]], *, limit: int = _SEARCH_RERANK_HOUSE_CONTEXT_LIMIT) -> None:
    if state is None:
        return
    compact: list[HouseLite] = []
    seen_house_ids: set[str] = set()
    for row in rows[:limit]:
        house = _house_context_row_to_lite(row)
        if house is None or house.house_id in seen_house_ids:
            continue
        seen_house_ids.add(house.house_id)
        compact.append(house)
        if len(compact) >= limit:
            break
    setattr(state, "house_context_top10", compact)


def _build_house_context_top10_rows(
    state: Any,
    fallback_candidates: list[Any] | None,
    *,
    tool_results: list[dict[str, Any]] | None = None,
    limit: int = _SEARCH_RERANK_HOUSE_CONTEXT_LIMIT,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_house_ids: set[str] = set()
    nearby_landmarks_by_community = _collect_nearby_landmarks_context(tool_results)

    def append_item(item: Any) -> bool:
        row = _house_context_row(item, nearby_landmarks_by_community=nearby_landmarks_by_community)
        house_id = row.get("house_id")
        if not isinstance(house_id, str) or not house_id or house_id in seen_house_ids:
            return False
        seen_house_ids.add(house_id)
        rows.append(row)
        return len(rows) >= limit

    sources: list[list[Any]] = []
    house_context_top10 = getattr(state, "house_context_top10", None)
    if isinstance(house_context_top10, list) and house_context_top10:
        sources.append(house_context_top10)

    last_candidates = getattr(state, "last_candidates", None)
    if isinstance(last_candidates, list) and last_candidates:
        sources.append(last_candidates)

    if isinstance(fallback_candidates, list) and fallback_candidates:
        sources.append(fallback_candidates)

    last_top5 = getattr(state, "last_top5", None)
    if isinstance(last_top5, list) and last_top5:
        sources.append(last_top5)

    for source in sources:
        for item in source:
            if append_item(item):
                return rows
    return rows


def _build_llm_compact_context_facts(state: Any, *, include_candidate_tags: bool) -> dict[str, Any]:
    if state is None:
        return {}

    facts: dict[str, Any] = {}
    case_type_value = getattr(getattr(state, "case_type", None), "value", None)
    if isinstance(case_type_value, str) and case_type_value:
        facts["case_type"] = case_type_value

    focus_house_id = getattr(getattr(state, "candidate_state", None), "focus_house_id", None) or getattr(state, "focus_house_id", None)
    if isinstance(focus_house_id, str) and focus_house_id:
        facts["focus_house_id"] = focus_house_id

    focus_platform = getattr(state, "focus_listing_platform", None)
    if isinstance(focus_platform, str) and focus_platform:
        facts["focus_listing_platform"] = focus_platform
    elif hasattr(focus_platform, "value") and isinstance(getattr(focus_platform, "value"), str):
        facts["focus_listing_platform"] = getattr(focus_platform, "value")

    constraints_source: dict[str, Any] = {}
    confirmed = getattr(state, "confirmed_constraints", None)
    if confirmed is not None and hasattr(confirmed, "model_dump"):
        confirmed_dict = confirmed.model_dump(exclude_none=True)
        if isinstance(confirmed_dict, dict):
            constraints_source = confirmed_dict
    req_hard = getattr(getattr(state, "req", None), "hard", None)
    if req_hard is not None and hasattr(req_hard, "model_dump"):
        hard_dict = req_hard.model_dump(exclude_none=True)
        if isinstance(hard_dict, dict):
            constraints_source = {**hard_dict, **constraints_source}
    constraints: dict[str, Any] = {}
    for key in _LLM_CONTEXT_HARD_KEYS:
        value = constraints_source.get(key)
        if value is None:
            continue
        if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
            constraints[key] = getattr(value, "value")
        else:
            constraints[key] = value
    if constraints:
        facts["constraints_summary"] = constraints

    soft_source: dict[str, Any] = {}
    soft = getattr(state, "soft_preferences", None)
    if soft is not None and hasattr(soft, "model_dump"):
        soft_dict = soft.model_dump(exclude_none=True)
        if isinstance(soft_dict, dict):
            soft_source = soft_dict
    soft_picked: dict[str, Any] = {}
    for key in _LLM_CONTEXT_SOFT_KEYS:
        value = soft_source.get(key)
        if value is None:
            continue
        soft_picked[key] = value
    if soft_picked:
        facts["soft_summary"] = soft_picked

    req_soft = getattr(getattr(state, "req", None), "soft", None)
    if req_soft is not None and hasattr(req_soft, "model_dump"):
        req_soft_dict = req_soft.model_dump(exclude_none=True)
        if isinstance(req_soft_dict, dict):
            compact_tag_need = _compact_tag_need_payload(req_soft_dict.get("tag_need_accumulated"))
            if any(compact_tag_need.values()):
                facts["tag_need_accumulated"] = compact_tag_need

    candidate_state = getattr(state, "candidate_state", None)
    candidate_ids: list[str] = []
    if candidate_state is not None:
        latest_ids = getattr(candidate_state, "latest_house_ids", None)
        if isinstance(latest_ids, list):
            for house_id in latest_ids:
                if not isinstance(house_id, str) or not house_id:
                    continue
                candidate_ids.append(house_id)
                if len(candidate_ids) >= _LLM_CONTEXT_HOUSE_LIMIT:
                    break
    if candidate_ids:
        facts["latest_house_ids"] = candidate_ids

    if include_candidate_tags:
        house_context_top10 = _build_house_context_top10_rows(state, [], limit=_SEARCH_RERANK_HOUSE_CONTEXT_LIMIT)
        if house_context_top10:
            facts["house_context_top10"] = house_context_top10

    pruned = _prune_empty_for_llm(facts, drop_false_keys=_SOFT_BOOL_DEFAULT_FALSE_KEYS)
    return pruned if isinstance(pruned, dict) else {}


def _build_llm_plan_context_facts(state: Any) -> dict[str, Any]:
    return _build_llm_compact_context_facts(state, include_candidate_tags=True)


def _build_llm_respond_context_facts(state: Any) -> dict[str, Any]:
    return _build_llm_compact_context_facts(state, include_candidate_tags=False)


def _sanitize_llm_parse(parsed: dict[str, Any]) -> dict[str, Any]:
    intent = _sanitize_plan_intent(parsed.get("intent"))
    if intent is None:
        return {}
    params = _sanitize_plan_params(parsed.get("params"))
    tag_need = _sanitize_plan_tag_need(parsed.get("tag_need"))
    return {"intent": intent, "params": params, "tag_need": tag_need}


def _sanitize_plan_intent(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    intent = value.strip().lower()
    allowed = {"chat", "search", "compare", "house_detail", "amenities", "listings", "rent_check", "rent", "terminate", "offline"}
    if intent in allowed:
        return intent
    return None


def _normalize_bedrooms_param(value: str) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    normalized = cleaned.translate(str.maketrans({"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5"}))
    numbers = re.findall(r"[1-9]", normalized)
    if not numbers:
        return None
    unique: list[str] = []
    seen: set[str] = set()
    for item in numbers:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return ",".join(unique) if unique else None


def _sanitize_plan_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    params: dict[str, Any] = {}

    district = value.get("district")
    if isinstance(district, str) and district.strip():
        params["district"] = district.strip()

    bedrooms = value.get("bedrooms")
    if isinstance(bedrooms, str):
        normalized_bedrooms = _normalize_bedrooms_param(bedrooms)
        if normalized_bedrooms:
            params["bedrooms"] = normalized_bedrooms

    for key in ("min_price", "max_price", "max_subway_dist", "min_area"):
        parsed = _coerce_int(value.get(key))
        if parsed is not None:
            params[key] = parsed

    decoration = value.get("decoration")
    if isinstance(decoration, str) and decoration.strip():
        params["decoration"] = decoration.strip()

    rental_type = value.get("rental_type")
    if isinstance(rental_type, str):
        cleaned_type = rental_type.strip()
        if cleaned_type in {"整租", "合租"}:
            params["rental_type"] = cleaned_type

    elevator = _coerce_bool(value.get("elevator"))
    if elevator is not None:
        params["elevator"] = elevator

    return params


def _sanitize_plan_tag_need(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {"must": [], "avoid": [], "prefer": []}
    output: dict[str, list[str]] = {"must": [], "avoid": [], "prefer": []}
    for key in ("must", "avoid", "prefer"):
        raw = value.get(key)
        if not isinstance(raw, list):
            continue
        seen: set[str] = set()
        tokens: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= 12:
                break
        output[key] = tokens
    return output


def _apply_tool_argument_aliases(operation_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    alias_map = _TOOL_ARGUMENT_KEY_ALIASES.get(operation_id)
    if not alias_map:
        return arguments

    remapped: dict[str, Any] = {}
    for key, value in arguments.items():
        if not isinstance(key, str):
            continue
        canonical = alias_map.get(key, key)
        if canonical not in remapped:
            remapped[canonical] = value
    return remapped


def _coerce_argument_value(value: Any, spec: dict[str, Any]) -> Any:
    type_name = str(spec.get("type", "string")).lower()
    param_name = spec.get("name")
    if type_name == "integer":
        coerced = _coerce_int(value)
    elif type_name == "number":
        coerced = _coerce_float(value)
    elif type_name == "boolean":
        coerced = _coerce_bool(value)
    elif type_name == "string":
        if value is None:
            return None
        coerced = str(value).strip()
        if not coerced:
            return None
        if param_name == "house_id":
            upper = coerced.upper()
            if not re.fullmatch(r"[A-Z]{2,4}_?\d{1,8}", upper):
                return None
            coerced = upper
    else:
        # Unknown schema types are kept as-is to avoid accidental data loss.
        coerced = value

    enum_values = spec.get("enum")
    if isinstance(enum_values, list) and enum_values:
        coerced = _normalize_enum_value(coerced, enum_values, spec.get("name"))
        if coerced not in enum_values:
            return None
    return coerced


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        compact = stripped.replace(",", "").replace("，", "")
        if compact.endswith("元"):
            compact = compact[:-1]
        if compact.endswith("块"):
            compact = compact[:-1]

        unit_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kK千wW万])", compact)
        if unit_match:
            base = float(unit_match.group(1))
            unit = unit_match.group(2).lower()
            if unit in {"k", "千"}:
                return int(base * 1000)
            if unit in {"w", "万"}:
                return int(base * 10000)

        cn_value = _coerce_cn_number(compact)
        if cn_value is not None:
            return cn_value

        try:
            return int(float(compact))
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "是"}:
            return True
        if normalized in {"false", "0", "no", "n", "否"}:
            return False
    return None


def _normalize_enum_value(value: Any, enum_values: list[Any], param_name: Any) -> Any:
    if value in enum_values:
        return value
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    if normalized in enum_values:
        return normalized

    if isinstance(param_name, str) and param_name == "listing_platform":
        if "58" in normalized and "58同城" in enum_values:
            return "58同城"
        if normalized in {"lianjia", "链家网"} and "链家" in enum_values:
            return "链家"
        if normalized.lower() == "anjuke" and "安居客" in enum_values:
            return "安居客"
    return value


def _coerce_cn_number(text: str) -> int | None:
    if not text or not re.search(r"[一二两三四五六七八九十百千万零〇]", text):
        return None
    if not re.fullmatch(r"[一二两三四五六七八九十百千万零〇]+", text):
        return None

    total = 0
    section = 0
    number = 0
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
            continue
        if ch in _CN_SMALL_UNITS:
            unit = _CN_SMALL_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
            continue
        if ch == "万":
            section += number
            if section == 0:
                section = 1
            total += section * 10000
            section = 0
            number = 0
            continue
        return None

    result = total + section + number
    return result if result > 0 else None


def _build_llm_context_facts(state: Any) -> dict[str, Any]:
    if state is None:
        return {}

    facts: dict[str, Any] = {}
    case_type = getattr(state, "case_type", None)
    case_type_value = getattr(case_type, "value", case_type)
    if isinstance(case_type_value, str) and case_type_value:
        facts["case_type"] = case_type_value

    focus_house_id = getattr(state, "focus_house_id", None)
    if isinstance(focus_house_id, str) and focus_house_id:
        facts["focus_house_id"] = focus_house_id

    focus_platform = getattr(state, "focus_listing_platform", None)
    if isinstance(focus_platform, str) and focus_platform:
        facts["focus_listing_platform"] = focus_platform
    elif hasattr(focus_platform, "value") and isinstance(getattr(focus_platform, "value"), str):
        facts["focus_listing_platform"] = getattr(focus_platform, "value")

    confirmed = getattr(state, "confirmed_constraints", None)
    if confirmed is not None and hasattr(confirmed, "model_dump"):
        confirmed_dict = confirmed.model_dump(exclude_none=True)
        picked = {}
        for key in (
            "district",
            "area",
            "community",
            "landmark_id",
            "landmark_name",
            "budget_min",
            "budget_max",
            "layout",
            "rent_type",
            "max_subway_dist",
            "max_distance",
            "max_commute_min",
            "utilities_type",
            "listing_platform",
        ):
            value = confirmed_dict.get(key)
            if value is None:
                continue
            if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
                picked[key] = getattr(value, "value")
            else:
                picked[key] = value
        if picked:
            facts["confirmed_constraints"] = picked

    soft = getattr(state, "soft_preferences", None)
    if soft is not None and hasattr(soft, "model_dump"):
        soft_dict = soft.model_dump(exclude_none=True)
        picked_soft: dict[str, Any] = {}
        for key in (
            "orientation",
            "decoration",
            "elevator",
            "noise_preference",
            "amenities",
            "preferred_tags",
            "avoid_tags",
            "value_for_money",
            "prefer_spacious",
            "prioritize_subway_distance",
            "prioritize_commute",
        ):
            value = soft_dict.get(key)
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            picked_soft[key] = value
        if picked_soft:
            facts["soft_preferences"] = picked_soft

    req_state = getattr(state, "req", None)
    if req_state is not None and hasattr(req_state, "model_dump"):
        req_dict = req_state.model_dump(exclude_none=True)
        if isinstance(req_dict, dict):
            facts["req"] = req_dict

    candidate_state = getattr(state, "candidate_state", None)
    if candidate_state is not None and hasattr(candidate_state, "model_dump"):
        candidate_dict = candidate_state.model_dump(exclude_none=True)
        if isinstance(candidate_dict, dict) and candidate_dict:
            facts["candidate_state"] = candidate_dict

    tag_lexicon = getattr(state, "tag_lexicon", None)
    if isinstance(tag_lexicon, dict) and tag_lexicon:
        compact_lexicon = dict(list(tag_lexicon.items())[:40])
        facts["tag_lexicon"] = compact_lexicon

    search_history = getattr(state, "search_history", None)
    if isinstance(search_history, list) and search_history:
        recent_searches: list[dict[str, Any]] = []
        for snap in search_history[-3:]:
            if not hasattr(snap, "house_ids"):
                continue
            house_ids = getattr(snap, "house_ids", [])
            if not isinstance(house_ids, list) or not house_ids:
                continue
            row: dict[str, Any] = {"house_ids": [str(x) for x in house_ids[:2]]}
            for key in ("district", "area", "community", "landmark_name"):
                value = getattr(snap, key, None)
                if isinstance(value, str) and value:
                    row[key] = value
            recent_searches.append(row)
        if recent_searches:
            facts["recent_searches"] = recent_searches
            facts["latest_search_house_ids"] = recent_searches[-1]["house_ids"]

    last_top5 = getattr(state, "last_top5", None)
    if isinstance(last_top5, list) and last_top5:
        compact_houses: list[dict[str, Any]] = []
        for item in last_top5[:3]:
            compact = _compact_house_for_llm_context(item)
            if compact:
                compact_houses.append(compact)
        if compact_houses:
            facts["latest_top_houses"] = compact_houses

    recent_turns = getattr(state, "recent_turns", None)
    if isinstance(recent_turns, list) and recent_turns:
        actions: list[dict[str, Any]] = []
        for turn in recent_turns[-6:]:
            intent = getattr(turn, "intent", None)
            if intent is None:
                continue
            intent_value = getattr(intent, "value", intent)
            if intent_value not in {"rent", "terminate", "offline"}:
                continue
            house_ids = getattr(turn, "house_ids", [])
            if not isinstance(house_ids, list):
                continue
            actions.append({"intent": intent_value, "house_ids": [str(x) for x in house_ids[:2]]})
        if actions:
            facts["recent_actions"] = actions

    return facts


def _compact_house_for_llm_context(item: Any) -> dict[str, Any]:
    house_id = getattr(item, "house_id", None)
    if not isinstance(house_id, str) or not house_id:
        return {}

    row: dict[str, Any] = {"house_id": house_id}
    for key in ("rent", "subway_distance", "commute_to_xierqi_min"):
        value = getattr(item, key, None)
        if value is None:
            continue
        row[key] = value

    pet_friendly = getattr(item, "pet_friendly", None)
    if isinstance(pet_friendly, bool):
        row["pet_friendly"] = pet_friendly
    else:
        tags = getattr(item, "tags", None)
        inferred = _infer_pet_friendly_from_tags(tags if isinstance(tags, list) else [])
        if inferred is not None:
            row["pet_friendly"] = inferred

    amenity_summary = getattr(item, "amenity_summary", None)
    if isinstance(amenity_summary, dict):
        compact_amenity: dict[str, Any] = {}
        for key in ("shopping_count", "nearest_shopping_m", "park_count", "nearest_park_m"):
            value = amenity_summary.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                compact_amenity[key] = value
        if compact_amenity:
            row["amenity_summary"] = compact_amenity

    tags = getattr(item, "tags", None)
    if isinstance(tags, list) and tags:
        key_tags = _pick_key_tags_for_llm(tags)
        if key_tags:
            row["key_tags"] = key_tags

    return row


def _infer_pet_friendly_from_tags(tags: list[Any]) -> bool | None:
    normalized = [str(tag).strip().lower() for tag in tags if isinstance(tag, str) and str(tag).strip()]
    if not normalized:
        return None
    if any(any(token in tag for token in ("不可养宠", "禁止养宠", "不允许养宠")) for tag in normalized):
        return False
    if any(any(token in tag for token in ("可养宠", "可养猫", "可养狗", "宠物友好", "仅限小型犬")) for tag in normalized):
        return True
    return None


def _pick_key_tags_for_llm(tags: list[Any], *, limit: int = 4) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    keywords = (
        "可养",
        "宠物",
        "公园",
        "商超",
        "地铁",
        "通勤",
        "采光",
        "朝",
        "电梯",
        "安静",
        "临街",
        "中介费",
        "网费",
        "物业费",
        "水电费",
        "取暖费",
    )
    for raw in tags:
        if not isinstance(raw, str):
            continue
        tag = raw.strip()
        if not tag or tag in seen:
            continue
        if any(word in tag for word in keywords):
            seen.add(tag)
            picked.append(tag)
        if len(picked) >= limit:
            break
    return picked


def _should_skip_llm_nlu(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return True
    if re.fullmatch(r"(你好|您好|hi|hello|hey|哈喽|在吗|在不在)[!！。,. ]*", text):
        return True
    if len(text) <= 8 and any(token in text for token in ("谢谢", "好的", "收到", "明白")):
        return True
    return False


def _collect_house_ids_from_state(state: Any) -> set[str]:
    house_ids: set[str] = set()
    if state is None:
        return house_ids

    candidate_state = getattr(state, "candidate_state", None)
    latest_ids = getattr(candidate_state, "latest_house_ids", None)
    if isinstance(latest_ids, list) and latest_ids:
        for house_id in latest_ids:
            if isinstance(house_id, str) and house_id:
                house_ids.add(house_id)
        return house_ids

    house_context_top10 = getattr(state, "house_context_top10", None)
    if isinstance(house_context_top10, list):
        for item in house_context_top10[:_SEARCH_RERANK_HOUSE_CONTEXT_LIMIT]:
            house_id = _value_from_item(item, "house_id")
            if isinstance(house_id, str) and house_id:
                house_ids.add(house_id)
        if house_ids:
            return house_ids

    last_top5 = getattr(state, "last_top5", None)
    if isinstance(last_top5, list):
        for item in last_top5[:5]:
            house_id = getattr(item, "house_id", None)
            if isinstance(house_id, str) and house_id:
                house_ids.add(house_id)
    return house_ids


def _collect_house_ids_from_tool_results(tool_results: list[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for row in tool_results:
        if not isinstance(row, dict) or row.get("success") is not True:
            continue
        output = row.get("output")
        _walk_house_ids(output, found)
    return found


def _walk_house_ids(payload: Any, found: set[str]) -> None:
    if isinstance(payload, dict):
        house_id = payload.get("house_id")
        if isinstance(house_id, str) and house_id:
            found.add(house_id)
        for value in payload.values():
            _walk_house_ids(value, found)
        return
    if isinstance(payload, list):
        for item in payload:
            _walk_house_ids(item, found)


_HOUSE_CONTEXT_REF_TOKENS = ("这套", "第一套", "第二套", "上面那个", "刚才推荐的")
_HOUSE_CONTEXT_ACTION_TOKENS = ("租", "能租", "看房", "vr", "带宠物", "宽带", "暖气", "物业", "押", "费用")


def _is_housing_context_inquiry(message: str, state: Any) -> bool:
    candidate_state = getattr(state, "candidate_state", None)
    latest_ids = getattr(candidate_state, "latest_house_ids", None)
    if not isinstance(latest_ids, list) or not latest_ids:
        return False

    text = (message or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in _HOUSE_CONTEXT_REF_TOKENS):
        return True
    if any(token in text for token in _HOUSE_CONTEXT_ACTION_TOKENS):
        return True
    return False


def _build_llm_nlu_messages(message: str, summary: str, context_facts: dict[str, Any] | None = None) -> list[dict[str, str]]:
    summary_text = summary[:220] if summary else ""
    context_text = _format_context_facts_for_prompt(context_facts)
    content = (
        "你是租房智能Agent决策器（Plan模块）。\n"
        "只输出1行字符串，不要解释。\n"
        "格式：i=<intent>|p=k:v;k:v|t=m:x,y;a:u,v;p:q,r\n"
        "intent可选：chat/search/compare/house_detail/amenities/listings/rent_check/rent/terminate/offline。\n"
        "p仅允许键：district,bedrooms,min_price,max_price,decoration,max_subway_dist,rental_type,min_area,elevator。\n"
        "t里 m=must, a=avoid, p=prefer；没值留空。\n"
        "规则：用户问“这套可租吗/我可以租吗/能租吗”时 i=rent_check；预算表达“左右/上下/附近/约/大概”时尽量给 min_price 与 max_price 区间。\n"
        "示例：i=search|p=district:朝阳;bedrooms:2;max_price:3500|t=m:;a:;p:仅线上VR看房\n"
        f"会话摘要：{summary_text}\n"
        f"上下文：\n{context_text}\n"
        f"用户输入：{message}"
    )
    return [{"role": "user", "content": content}]


@lru_cache(maxsize=1)
def _load_llm_tool_names() -> tuple[str, ...]:
    tools = _load_llm_tools()
    names: list[str] = []
    for item in tools:
        function_node = item.get("function")
        if not isinstance(function_node, dict):
            continue
        name = function_node.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)


def _normalize_messages_for_eval(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages:
        return [{"role": "user", "content": ""}]

    for item in reversed(messages):
        role = str(item.get("role", "")).strip().lower()
        content = item.get("content")
        if role == "user" and isinstance(content, str):
            return [{"role": "user", "content": content}]

    for item in reversed(messages):
        content = item.get("content")
        if isinstance(content, str):
            return [{"role": "user", "content": content}]

    return [{"role": "user", "content": ""}]


def _extract_llm_assistant_message(data: dict[str, Any]) -> dict[str, Any] | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    return message


def _extract_llm_text_content(data: dict[str, Any]) -> str | None:
    message = _extract_llm_assistant_message(data)
    if not message:
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


def _extract_llm_tool_plan(data: dict[str, Any]) -> dict[str, Any] | None:
    message = _extract_llm_assistant_message(data)
    if not message:
        return None
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None
    function_node = first_call.get("function")
    if not isinstance(function_node, dict):
        return None

    operation_id = function_node.get("name")
    if not isinstance(operation_id, str) or not operation_id:
        return None

    arguments = _parse_llm_tool_arguments(function_node.get("arguments"))
    parsed: dict[str, Any] = {
        "tool_plan": {
            "operationId": operation_id,
            "arguments": arguments,
        },
        "confidence": 0.9,
    }
    mapped_intent = _load_operation_intents().get(operation_id)
    if isinstance(mapped_intent, str) and mapped_intent:
        parsed["intent"] = mapped_intent
    return parsed


def _parse_llm_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if isinstance(k, str)}
    if not isinstance(raw, str):
        return {}

    parsed = _extract_json_object(raw)
    if isinstance(parsed, dict):
        return parsed

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if isinstance(k, str)}
    return {}


def _parse_llm_nlu_payload(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        text = text.replace("```text", "").replace("```txt", "").replace("```", "").strip()
    text = text.replace("｜", "|").replace("：", ":")

    compact_parsed = _parse_llm_nlu_compact_payload(text)
    if compact_parsed:
        return compact_parsed

    parsed = _extract_json_object(text)
    if isinstance(parsed, dict):
        return parsed
    return None


def _parse_llm_nlu_compact_payload(text: str) -> dict[str, Any] | None:
    segments = [seg.strip() for seg in re.split(r"[|\n]+", text) if isinstance(seg, str) and seg.strip()]
    if not segments:
        return None

    parsed: dict[str, Any] = {"params": {}, "tag_need": {"must": [], "avoid": [], "prefer": []}}
    has_signal = False

    for segment in segments:
        key, value = _split_compact_pair(segment)
        if key is None:
            continue
        norm_key = key.strip().lower()
        if norm_key in {"i", "intent"}:
            if value:
                parsed["intent"] = value.strip().lower()
                has_signal = True
            continue
        if norm_key in {"p", "params"}:
            params = _parse_llm_nlu_compact_params(value)
            if params:
                parsed["params"].update(params)
                has_signal = True
            continue
        if norm_key in {"t", "tag", "tag_need"}:
            tag_need = _parse_llm_nlu_compact_tag_need(value)
            if any(tag_need.values()):
                for k in ("must", "avoid", "prefer"):
                    parsed["tag_need"][k].extend(tag_need[k])
                has_signal = True
            continue

        param_key = _LLM_NLU_PARAM_KEY_ALIASES.get(norm_key)
        if param_key:
            cleaned = (value or "").strip()
            if cleaned:
                parsed["params"][param_key] = cleaned
                has_signal = True
            continue

        tag_key = _LLM_NLU_TAG_KEY_ALIASES.get(norm_key)
        if tag_key:
            tokens = _split_compact_tag_tokens(value)
            if tokens:
                parsed["tag_need"][tag_key].extend(tokens)
                has_signal = True

    if not has_signal:
        return None
    return parsed


def _split_compact_pair(segment: str) -> tuple[str | None, str]:
    if ":" in segment and "=" in segment:
        eq_idx = segment.find("=")
        colon_idx = segment.find(":")
        if eq_idx < colon_idx:
            key, value = segment.split("=", 1)
            return key, value
        key, value = segment.split(":", 1)
        return key, value
    if "=" in segment:
        key, value = segment.split("=", 1)
        return key, value
    if ":" in segment:
        key, value = segment.split(":", 1)
        return key, value
    return None, ""


def _parse_llm_nlu_compact_params(raw: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in re.findall(r"([a-zA-Z_]+)\s*[:=]\s*([^;|]+)", raw or ""):
        norm_key = _LLM_NLU_PARAM_KEY_ALIASES.get(key.strip().lower())
        if not norm_key:
            continue
        cleaned = value.strip().strip("\"'")
        if not cleaned or cleaned.lower() in {"none", "null"}:
            continue
        params[norm_key] = cleaned
    return params


def _parse_llm_nlu_compact_tag_need(raw: str) -> dict[str, list[str]]:
    output = {"must": [], "avoid": [], "prefer": []}
    for key, value in re.findall(r"([a-zA-Z_]+)\s*[:=]\s*([^;|]*)", raw or ""):
        norm_key = _LLM_NLU_TAG_KEY_ALIASES.get(key.strip().lower())
        if not norm_key:
            continue
        output[norm_key].extend(_split_compact_tag_tokens(value))
    return output


def _split_compact_tag_tokens(raw: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for item in re.split(r"[,，、/]+", raw or ""):
        token = item.strip()
        if not token or token in seen or token in {"-", "无", "none", "null"}:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


async def _analyze_intent_with_llm(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    session_id: str,
    message: str,
    summary: str,
    context_facts: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    response_data = await _forward_chat_completion(
        http_client,
        model_ip=model_ip,
        messages=_build_llm_nlu_messages(message, summary, context_facts=context_facts),
        tools=_load_llm_tools(),
        session_id=session_id,
        step=STEP_LLM_NLU,
        llm_stage="plan",
    )
    text_parse: dict[str, Any] | None = None
    llm_text = _extract_llm_text_content(response_data)
    if llm_text:
        parsed = _parse_llm_nlu_payload(llm_text)
        if isinstance(parsed, dict):
            text_parse = parsed
    if isinstance(text_parse, dict):
        sanitized = _sanitize_llm_parse(text_parse)
        if sanitized:
            return sanitized
    return None


def _merge_llm_parse_candidates(
    tool_parse: dict[str, Any] | None,
    text_parse: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if tool_parse is None and text_parse is None:
        return None
    if tool_parse is None:
        return text_parse
    if text_parse is None:
        return tool_parse

    merged = dict(text_parse)
    # Function-call arguments are the most executable form; keep them authoritative.
    merged["tool_plan"] = tool_parse.get("tool_plan", merged.get("tool_plan"))
    if "intent" not in merged and "intent" in tool_parse:
        merged["intent"] = tool_parse["intent"]
    if "confidence" not in merged and "confidence" in tool_parse:
        merged["confidence"] = tool_parse["confidence"]
    return merged


def _build_search_rerank_house_context_top10(
    state: Any,
    fallback_candidates: list[Any],
    *,
    tool_results: list[dict[str, Any]] | None = None,
    limit: int = _SEARCH_RERANK_HOUSE_CONTEXT_LIMIT,
) -> list[dict[str, Any]]:
    rows = _build_house_context_top10_rows(
        state,
        fallback_candidates,
        tool_results=tool_results,
        limit=limit,
    )
    if state is not None and rows:
        _set_state_house_context_top10(state, rows, limit=limit)
    return rows


def _build_llm_search_rerank_messages(
    *,
    user_message: str,
    draft_reply: str,
    context_facts: dict[str, Any] | None,
    house_context_top10: list[dict[str, Any]],
) -> list[dict[str, str]]:
    context_text = _format_context_facts_for_prompt(context_facts)
    house_context_json = (
        json.dumps(house_context_top10[:_SEARCH_RERANK_HOUSE_CONTEXT_LIMIT], ensure_ascii=False, separators=(",", ":"))
        if house_context_top10
        else "[]"
    )
    content = (
        "你是租房结果重排器。\n"
        "输入包含房源上下文（最多10套）。请根据用户输入和上下文筛出最相关5套。\n"
        "只输出1行字符串：m=<中文结论>|h=<house_id1,house_id2,house_id3,house_id4,house_id5>\n"
        "规则：\n"
        "1) h 里的 house_id 必须全部来自房源上下文。\n"
        "2) h 最多5个，按推荐优先级排序。\n"
        "3) m 直接对用户说结论，中文，简洁。\n"
        "4) 只输出一行，不要JSON、不要多余解释。\n"
        f"用户输入：{user_message}\n"
        f"当前草稿回复：{draft_reply}\n"
        f"上下文：\n{context_text}\n"
        f"房源上下文top10(JSON)：{house_context_json}"
    )
    return [{"role": "user", "content": content}]


def _extract_house_ids_from_any(value: Any, *, max_count: int = 5) -> list[str]:
    items: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                items.append(item)
            elif item is not None:
                items.append(str(item))
    elif isinstance(value, str):
        items.append(value)
    elif value is not None:
        items.append(str(value))

    extracted: list[str] = []
    seen: set[str] = set()
    for item in items:
        for match in _HOUSE_ID_EXTRACT_PATTERN.findall(item):
            normalized = match.upper()
            if normalized in seen:
                continue
            seen.add(normalized)
            extracted.append(normalized)
            if len(extracted) >= max_count:
                return extracted
    return extracted


def _parse_llm_search_rerank_compact_payload(content: str) -> tuple[str | None, list[str]] | None:
    text = content.strip()
    if not text:
        return None

    message: str | None = None
    house_ids: list[str] = []
    for segment in text.split("|"):
        if "=" not in segment:
            continue
        key, raw_value = segment.split("=", 1)
        normalized_key = key.strip().lower()
        value = raw_value.strip()
        if normalized_key in {"m", "message"}:
            message = value or None
            continue
        if normalized_key in {"h", "houses", "ids", "house_ids"}:
            house_ids = _extract_house_ids_from_any(value, max_count=5)
    if message is None and not house_ids:
        return None
    return message, house_ids


def _extract_llm_search_rerank_result(data: dict[str, Any]) -> tuple[str | None, list[str]]:
    content = _extract_llm_text_content(data)
    if not isinstance(content, str) or not content.strip():
        return None, []
    stripped = content.strip()
    parsed = _extract_json_object(stripped)
    if isinstance(parsed, dict):
        raw_message = parsed.get("message")
        message = raw_message.strip() if isinstance(raw_message, str) and raw_message.strip() else None
        house_ids = _extract_house_ids_from_any(parsed.get("houses"), max_count=5)
        if not house_ids:
            house_ids = _extract_house_ids_from_any(parsed.get("house_ids"), max_count=5)
        if not house_ids:
            house_ids = _extract_house_ids_from_any(parsed.get("ids"), max_count=5)
        return message, house_ids

    compact = _parse_llm_search_rerank_compact_payload(stripped)
    if compact is not None:
        return compact

    house_ids = _extract_house_ids_from_any(stripped, max_count=5)
    return (stripped if stripped else None), house_ids


def _house_lite_to_view_model(item: HouseLite) -> HouseViewModel:
    return HouseViewModel(
        house_id=item.house_id,
        listing_platform=item.listing_platform,
        rent=item.rent,
        layout=item.layout,
        area=item.area,
        district=item.district,
        community=item.community,
        subway_distance=item.subway_distance,
        commute_to_xierqi_min=item.commute_to_xierqi_min,
        available_date=item.available_date,
        tags=item.tags,
    )


def _build_reranked_house_views(
    *,
    selected_house_ids: list[str],
    state: Any,
    fallback_views: list[HouseViewModel],
    limit: int = 5,
) -> list[HouseViewModel]:
    view_map: dict[str, HouseViewModel] = {}
    for view in fallback_views:
        if isinstance(view, HouseViewModel):
            view_map[view.house_id] = view

    lite_map: dict[str, HouseLite] = {}

    def remember_lite(item: Any) -> None:
        if isinstance(item, HouseLite):
            lite_map[item.house_id] = item
            return
        row = _house_context_row(item)
        lite = _house_context_row_to_lite(row)
        if isinstance(lite, HouseLite):
            lite_map[lite.house_id] = lite

    house_context_top10 = getattr(state, "house_context_top10", None)
    if isinstance(house_context_top10, list):
        for item in house_context_top10:
            remember_lite(item)

    last_candidates = getattr(state, "last_candidates", None)
    if isinstance(last_candidates, list):
        for item in last_candidates:
            remember_lite(item)

    output: list[HouseViewModel] = []
    seen: set[str] = set()

    def append_house(house_id: str) -> None:
        if house_id in seen:
            return
        seen.add(house_id)
        if house_id in view_map:
            output.append(view_map[house_id])
            return
        lite = lite_map.get(house_id)
        if isinstance(lite, HouseLite):
            output.append(_house_lite_to_view_model(lite))

    for house_id in selected_house_ids:
        append_house(house_id)
        if len(output) >= limit:
            return output[:limit]

    for view in fallback_views:
        if isinstance(view, HouseViewModel):
            append_house(view.house_id)
        if len(output) >= limit:
            return output[:limit]
    return output[:limit]


def _merge_referenced_house_ids(debug: dict[str, Any], house_ids: list[str]) -> None:
    existing = debug.get("referenced_house_ids")
    merged: list[str] = []
    seen: set[str] = set()
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, str) and item and item not in seen:
                seen.add(item)
                merged.append(item)
    for item in house_ids:
        if isinstance(item, str) and item and item not in seen:
            seen.add(item)
            merged.append(item)
    if merged:
        debug["referenced_house_ids"] = merged[:10]


def _apply_search_rerank_to_state(
    state: Any,
    *,
    reranked_views: list[HouseViewModel],
    assistant_text: str,
    dialogue_manager: Any = None,
) -> None:
    if not reranked_views:
        return
    house_ids = [item.house_id for item in reranked_views if isinstance(item.house_id, str) and item.house_id]
    if not house_ids:
        return

    setattr(state, "last_top5", reranked_views[:5])
    candidate_state = getattr(state, "candidate_state", None)
    if candidate_state is not None:
        setattr(candidate_state, "latest_house_ids", house_ids[:5])
        setattr(candidate_state, "focus_house_id", house_ids[0])
    setattr(state, "focus_house_id", house_ids[0])

    context_rows = _build_house_context_top10_rows(state, reranked_views, limit=_SEARCH_RERANK_HOUSE_CONTEXT_LIMIT)
    if context_rows:
        row_map = {row.get("house_id"): row for row in context_rows if isinstance(row.get("house_id"), str)}
        reordered_rows: list[dict[str, Any]] = []
        seen_house_ids: set[str] = set()
        for house_id in house_ids[:5]:
            row = row_map.get(house_id)
            if isinstance(row, dict) and house_id not in seen_house_ids:
                seen_house_ids.add(house_id)
                reordered_rows.append(row)
        for row in context_rows:
            row_house_id = row.get("house_id")
            if not isinstance(row_house_id, str) or row_house_id in seen_house_ids:
                continue
            seen_house_ids.add(row_house_id)
            reordered_rows.append(row)
            if len(reordered_rows) >= _SEARCH_RERANK_HOUSE_CONTEXT_LIMIT:
                break
        _set_state_house_context_top10(state, reordered_rows, limit=_SEARCH_RERANK_HOUSE_CONTEXT_LIMIT)

    search_history = getattr(state, "search_history", None)
    if isinstance(search_history, list) and search_history:
        snapshot = search_history[-1]
        if hasattr(snapshot, "house_ids"):
            snapshot.house_ids = house_ids[:5]

    recent_turns = getattr(state, "recent_turns", None)
    if isinstance(recent_turns, list) and recent_turns:
        recent_turns[-1].house_ids = house_ids[:5]
        recent_turns[-1].assistant = preview_text(assistant_text, limit=180)

    if dialogue_manager is not None:
        platform_parser = getattr(dialogue_manager, "_platform_from_text", None)
        if callable(platform_parser):
            parsed_platform = platform_parser(reranked_views[0].listing_platform)
            if parsed_platform is not None:
                setattr(state, "focus_listing_platform", parsed_platform)
        summary_builder = getattr(dialogue_manager, "_build_conversation_summary", None)
        if callable(summary_builder):
            setattr(state, "conversation_summary", summary_builder(state))


async def _rerank_search_results_with_llm(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    session_id: str,
    user_message: str,
    draft_reply: str,
    context_facts: dict[str, Any] | None,
    house_context_top10: list[dict[str, Any]],
) -> tuple[str | None, list[str]]:
    response_data = await _forward_chat_completion(
        http_client,
        model_ip=model_ip,
        messages=_build_llm_search_rerank_messages(
            user_message=user_message,
            draft_reply=draft_reply,
            context_facts=context_facts,
            house_context_top10=house_context_top10,
        ),
        tools=[],
        session_id=session_id,
        step=STEP_LLM_SEARCH_RERANK,
        llm_stage="search_rerank",
    )
    return _extract_llm_search_rerank_result(response_data)


def _build_llm_detail_reply_messages(
    *,
    user_message: str,
    draft_reply: str,
    context_facts: dict[str, Any] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    context_text = _format_context_facts_for_prompt(context_facts)
    tool_results_text = _format_tool_results_for_prompt(tool_results)
    content = (
        "你是租房智能Agent的回复生成器。必须严格基于工具结果作答，禁止臆测。\n"
        "任务：根据“用户原话 + 工具结果 + Agent草稿”生成最终回复。\n"
        "硬性要求：\n"
        "1) 不得编造房源事实（状态、价格、距离、朝向等）。若工具结果缺失，明确说明“当前工具结果未提供该信息”。\n"
        "2) 若是操作类结果（租房/退租/下架），要明确说明是否成功及关键对象（house_id、平台、状态）。\n"
        "3) 语气专业、友善、简洁。\n"
        "4) 若上下文中有候选房源摘要或标签，可作为已知事实引用（这些来自上游工具结果的会话压缩）。\n"
        "5) 仅输出JSON object，格式必须为：{\"assistant_reply\":\"...\"}，不要输出其他字段。\n"
        f"用户原话：{user_message}\n"
        f"Agent草稿：{draft_reply}\n"
        f"会话上下文：\n{context_text}\n"
        f"工具结果摘要：\n{tool_results_text}"
    )
    return [{"role": "user", "content": content}]


def _extract_llm_assistant_reply(data: dict[str, Any]) -> str | None:
    message = _extract_llm_assistant_message(data)
    if not message:
        return None

    content = message.get("content")
    if not isinstance(content, str):
        return None

    stripped = content.strip()
    if not stripped:
        return None

    parsed = _extract_json_object(stripped)
    if isinstance(parsed, dict):
        reply = parsed.get("assistant_reply")
        if isinstance(reply, str) and reply.strip():
            return reply.strip()
        message = parsed.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return stripped


async def _polish_detail_reply_with_llm(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    session_id: str,
    user_message: str,
    draft_reply: str,
    context_facts: dict[str, Any] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> str | None:
    response_data = await _forward_chat_completion(
        http_client,
        model_ip=model_ip,
        messages=_build_llm_detail_reply_messages(
            user_message=user_message,
            draft_reply=draft_reply,
            context_facts=context_facts,
            tool_results=tool_results,
        ),
        tools=[],
        session_id=session_id,
        step=STEP_LLM_RESPOND,
        llm_stage="respond",
    )
    return _extract_llm_assistant_reply(response_data)


async def _preload_landmark_catalog(*, cache: CacheManager, landmarks_client: LandmarksClient) -> None:
    started = time.perf_counter()
    try:
        landmarks = await landmarks_client.list_landmarks()
    except DataSourceError as exc:
        log_event(
            LOGGER,
            "startup.landmarks.preload.failed",
            error=str(exc),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return

    stats: dict[str, Any] | None = None
    try:
        raw_stats = await landmarks_client.stats()
        if isinstance(raw_stats, dict):
            stats = raw_stats
    except DataSourceError:
        stats = None

    cache.prime_landmark_catalog(landmarks, stats)
    log_event(
        LOGGER,
        "startup.landmarks.preload.done",
        landmark_count=len(landmarks),
        landmark_alias_count=len(cache.landmark_name_aliases),
        district_alias_count=len(cache.landmark_district_aliases),
        category_count=len(cache.landmark_categories),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


def create_app(settings: AgentSettings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    _configure_search_rerank_context_settings(cfg)
    setup_logging(cfg.log_level)
    # Fail fast at startup if tool schema is unavailable or invalid.
    _load_llm_tools()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(
            connect=cfg.timeout.connect,
            read=cfg.timeout.read,
            write=cfg.timeout.write,
            pool=cfg.timeout.pool,
        )
        limits = httpx.Limits(
            max_connections=cfg.limits.max_connections,
            max_keepalive_connections=cfg.limits.max_keepalive_connections,
        )
        http_client = httpx.AsyncClient(timeout=timeout, limits=limits)

        cache = CacheManager(cfg)
        state_store = StateStore(cfg)
        landmarks_client = LandmarksClient(cfg.api_base_url, cfg.default_user_id, http_client)
        houses_client = HousesClient(cfg.api_base_url, cfg.default_user_id, http_client)
        startup_tokens = bind_log_context(
            trace_id="startup_landmarks",
            session_id=STARTUP_LANDMARK_PRELOAD_SESSION_ID,
            case_type="system",
            user_id=cfg.default_user_id,
        )
        try:
            await _preload_landmark_catalog(cache=cache, landmarks_client=landmarks_client)
        finally:
            reset_log_context(startup_tokens)
        service = AgentService(
            settings=cfg,
            state_store=state_store,
            landmarks_client=landmarks_client,
            houses_client=houses_client,
            cache=cache,
        )

        app.state.settings = cfg
        app.state.http_client = http_client
        app.state.agent_service = service
        app.state.session_model_ip: dict[str, str] = {}

        try:
            yield
        finally:
            await http_client.aclose()

    app = FastAPI(title="Smart Rental Agent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_http_requests(request: Request, call_next):
        started = time.perf_counter()
        raw_body = await request.body()

        async def receive() -> dict[str, bytes | bool]:
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request._receive = receive

        body_preview = raw_body.decode("utf-8", errors="replace") if raw_body else ""
        if len(body_preview) > 500:
            body_preview = body_preview[:500] + "...(truncated)"

        log_event(
            LOGGER,
            "http.request.in",
            step=STEP_HTTP,
            method=request.method,
            path=request.url.path,
            query=str(request.query_params),
            body=body_preview or "<empty>",
        )

        req_content_type = request.headers.get("content-type")
        http_session_id = _resolve_http_session_id(request, raw_body, req_content_type)
        should_log_http_io = request.method in {"GET", "POST"} and not _is_debug_agent_io_path(request.url.path)
        if should_log_http_io:
            log_json_event(
                HTTP_IO_LOGGER,
                {
                    **get_log_context(),
                    "event": "http.agent_io.request",
                    "session_id": http_session_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.query_params),
                    "request_content_type": req_content_type or "<unknown>",
                    "request_body": _parse_http_body_for_io(raw_body, req_content_type),
                },
            )

        response = await call_next(request)
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        replay_response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            LOGGER,
            "http.request.out",
            step=STEP_HTTP,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        if should_log_http_io:
            resp_content_type = replay_response.headers.get("content-type")
            log_json_event(
                HTTP_IO_LOGGER,
                {
                    **get_log_context(),
                    "event": "http.agent_io.response",
                    "session_id": http_session_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.query_params),
                    "status_code": replay_response.status_code,
                    "response_content_type": resp_content_type or "<unknown>",
                    "response_body": _parse_http_body_for_io(response_body, resp_content_type),
                    "duration_ms": duration_ms,
                },
            )
        return replay_response

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(req: InvokeRequest) -> InvokeResponse:
        started = time.perf_counter()
        trace_id = uuid4().hex[:12]
        user_id = req.user_id or cfg.default_user_id
        tokens = bind_log_context(
            trace_id=trace_id,
            session_id=req.session_id,
            case_type=req.case_type.value,
            user_id=user_id,
        )
        try:
            log_event(
                LOGGER,
                "invoke.received",
                step=STEP_RECV_USER_QUERY,
                message=preview_text(req.message, limit=300),
                history_len=len(req.history),
                meta_keys=sorted(req.meta.keys()),
            )
            service: AgentService = app.state.agent_service
            resp = await service.handle(req)
            log_event(
                LOGGER,
                "invoke.completed",
                step=STEP_FINAL_RESPONSE,
                duration_ms=int((time.perf_counter() - started) * 1000),
                candidate_count=len(resp.candidates),
                clarify_count=len(resp.clarify_questions),
                response=preview_text(resp.text, limit=300),
            )
            return resp
        except Exception as exc:
            log_event(
                LOGGER,
                "invoke.failed",
                step=STEP_FINAL_RESPONSE,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )
            raise
        finally:
            reset_log_context(tokens)

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        trace_id = uuid4().hex[:12]
        service: AgentService = app.state.agent_service
        existing_state = service.state_store.get(req.session_id) if hasattr(service, "state_store") else None
        resolved_case_type = CaseType.multi if existing_state is not None else CaseType.single
        tokens = bind_log_context(
            trace_id=trace_id,
            session_id=req.session_id,
            case_type=resolved_case_type.value,
            user_id=cfg.default_user_id,
        )
        tool_recorder_tokens = begin_tool_recording()
        try:
            log_event(
                LOGGER,
                "chat.received",
                step=STEP_RECV_USER_QUERY,
                model_ip=req.model_ip,
                message=preview_text(req.message, limit=300),
            )
            app.state.session_model_ip[req.session_id] = req.model_ip

            invoke_meta: dict[str, Any] = {"model_ip": req.model_ip}
            state_summary = ""
            context_facts: dict[str, Any] = {}
            state = existing_state
            if state is not None and getattr(state, "conversation_summary", ""):
                state_summary = state.conversation_summary
                context_facts = _build_llm_plan_context_facts(state)

            prompt_basis = req.message + state_summary + json.dumps(context_facts, ensure_ascii=False)
            nlu_prompt_tokens = (
                service.rough_token_estimate(prompt_basis)
                if hasattr(service, "rough_token_estimate")
                else max(1, int(len(prompt_basis) / 2))
            )
            if not _should_skip_llm_nlu(req.message):
                try:
                    http_client: httpx.AsyncClient = app.state.http_client
                    llm_parse = await _analyze_intent_with_llm(
                        http_client,
                        model_ip=req.model_ip,
                        session_id=req.session_id,
                        message=req.message,
                        summary=state_summary,
                        context_facts=context_facts,
                    )
                    if llm_parse:
                        invoke_meta["llm_parse"] = llm_parse
                        log_event(
                            LOGGER,
                            "chat.llm_nlu.applied",
                            step=STEP_LLM_NLU,
                            llm_parse=llm_parse,
                        )
                        if hasattr(service, "record_llm_fallback_usage"):
                            completion_tokens = (
                                service.rough_token_estimate(json.dumps(llm_parse, ensure_ascii=False))
                                if hasattr(service, "rough_token_estimate")
                                else max(1, int(len(json.dumps(llm_parse, ensure_ascii=False)) / 2))
                            )
                            service.record_llm_fallback_usage(req.session_id, nlu_prompt_tokens + completion_tokens)
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                    log_event(
                        LOGGER,
                        "chat.llm_nlu.failed",
                        step=STEP_LLM_NLU,
                        error=str(exc),
                    )
            else:
                log_event(LOGGER, "chat.llm_nlu.skipped", step=STEP_LLM_NLU, reason="simple_chat_message")

            log_event(
                LOGGER,
                "chat.agent.invoke.start",
                step=STEP_AGENT_PIPELINE,
                input_message=preview_text(req.message, limit=300),
            )
            invoke_resp = await service.handle(
                InvokeRequest(
                    session_id=req.session_id,
                    case_type=resolved_case_type,
                    message=req.message,
                    meta=invoke_meta,
                )
            )
            log_event(
                LOGGER,
                "chat.agent.invoke.done",
                step=STEP_AGENT_PIPELINE,
                invoke_text=preview_text(invoke_resp.text, limit=300),
                candidate_count=len(invoke_resp.candidates),
                clarify_count=len(invoke_resp.clarify_questions),
                debug=invoke_resp.debug,
            )

            output_text = invoke_resp.text
            response_kind = str(invoke_resp.debug.get("response_kind", "chat"))
            llm_parse = invoke_meta.get("llm_parse") if isinstance(invoke_meta.get("llm_parse"), dict) else {}
            tool_results = get_tool_results()
            latest_state = service.state_store.get(req.session_id) if hasattr(service, "state_store") else None
            respond_context_facts = _build_llm_respond_context_facts(latest_state) if latest_state is not None else context_facts
            intent_value = str(invoke_resp.debug.get("intent", "")).strip().lower()
            if response_kind in {"detail", "action"} and tool_results:
                respond_prompt_basis = (
                    req.message
                    + invoke_resp.text
                    + json.dumps(respond_context_facts or {}, ensure_ascii=False)
                    + json.dumps(tool_results, ensure_ascii=False)
                )
                respond_prompt_tokens = (
                    service.rough_token_estimate(respond_prompt_basis)
                    if hasattr(service, "rough_token_estimate")
                    else max(1, int(len(respond_prompt_basis) / 2))
                )
                try:
                    polished_reply = await _polish_detail_reply_with_llm(
                        app.state.http_client,
                        model_ip=req.model_ip,
                        session_id=req.session_id,
                        user_message=req.message,
                        draft_reply=invoke_resp.text,
                        context_facts=respond_context_facts,
                        tool_results=tool_results,
                    )
                    if isinstance(polished_reply, str) and polished_reply.strip():
                        output_text = polished_reply.strip()
                        log_event(
                            LOGGER,
                            "chat.llm_respond.applied",
                            step=STEP_LLM_RESPOND,
                            llm_output=preview_text(output_text, limit=300),
                        )
                        if hasattr(service, "record_llm_fallback_usage"):
                            completion_tokens = (
                                service.rough_token_estimate(output_text)
                                if hasattr(service, "rough_token_estimate")
                                else max(1, int(len(output_text) / 2))
                            )
                            service.record_llm_fallback_usage(req.session_id, respond_prompt_tokens + completion_tokens)
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                    log_event(
                        LOGGER,
                        "chat.llm_respond.failed",
                        step=STEP_LLM_RESPOND,
                        error=str(exc),
                    )
            if response_kind == "search" and latest_state is not None:
                house_context_top10 = _build_search_rerank_house_context_top10(
                    latest_state,
                    invoke_resp.candidates,
                    tool_results=tool_results,
                    limit=_SEARCH_RERANK_HOUSE_CONTEXT_LIMIT,
                )
                state_dirty = bool(house_context_top10)
                if 0 < len(house_context_top10) <= 5:
                    invoke_resp.debug["llm_search_rerank"] = {
                        "applied": False,
                        "reason": "candidate_count_lte_5",
                        "candidate_count": len(house_context_top10),
                    }
                    log_event(
                        LOGGER,
                        "chat.llm_search_rerank.skipped",
                        step=STEP_LLM_SEARCH_RERANK,
                        reason="candidate_count_lte_5",
                        candidate_count=len(house_context_top10),
                    )
                elif len(house_context_top10) >= 2:
                    rerank_prompt_basis = (
                        req.message
                        + invoke_resp.text
                        + json.dumps(respond_context_facts or {}, ensure_ascii=False)
                        + json.dumps(house_context_top10, ensure_ascii=False)
                    )
                    rerank_prompt_tokens = (
                        service.rough_token_estimate(rerank_prompt_basis)
                        if hasattr(service, "rough_token_estimate")
                        else max(1, int(len(rerank_prompt_basis) / 2))
                    )
                    try:
                        rerank_message, rerank_house_ids = await _rerank_search_results_with_llm(
                            app.state.http_client,
                            model_ip=req.model_ip,
                            session_id=req.session_id,
                            user_message=req.message,
                            draft_reply=invoke_resp.text,
                            context_facts=respond_context_facts,
                            house_context_top10=house_context_top10,
                        )
                        reranked_views = _build_reranked_house_views(
                            selected_house_ids=rerank_house_ids,
                            state=latest_state,
                            fallback_views=invoke_resp.candidates,
                            limit=5,
                        )
                        if reranked_views:
                            invoke_resp.candidates = reranked_views
                            reranked_ids = [item.house_id for item in reranked_views]
                            _merge_referenced_house_ids(invoke_resp.debug, reranked_ids)
                            invoke_resp.debug["llm_search_rerank"] = {
                                "applied": True,
                                "selected_house_ids": reranked_ids,
                                "candidate_count": len(house_context_top10),
                            }
                            if isinstance(rerank_message, str) and rerank_message.strip():
                                output_text = rerank_message.strip()

                            _apply_search_rerank_to_state(
                                latest_state,
                                reranked_views=reranked_views,
                                assistant_text=output_text,
                                dialogue_manager=getattr(service, "dialogue", None),
                            )
                            state_dirty = True
                            log_event(
                                LOGGER,
                                "chat.llm_search_rerank.applied",
                                step=STEP_LLM_SEARCH_RERANK,
                                selected_house_ids=reranked_ids,
                                message_preview=preview_text(output_text, limit=220),
                            )
                            if hasattr(service, "record_llm_fallback_usage"):
                                completion_basis = (rerank_message or "") + ",".join(reranked_ids)
                                completion_tokens = (
                                    service.rough_token_estimate(completion_basis)
                                    if hasattr(service, "rough_token_estimate")
                                    else max(1, int(len(completion_basis) / 2))
                                )
                                service.record_llm_fallback_usage(req.session_id, rerank_prompt_tokens + completion_tokens)
                    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                        log_event(
                            LOGGER,
                            "chat.llm_search_rerank.failed",
                            step=STEP_LLM_SEARCH_RERANK,
                            error=str(exc),
                        )
                if state_dirty and hasattr(service, "state_store"):
                    service.state_store.upsert(latest_state)
            house_intents = {"search", "compare", "house_detail", "amenities", "listings", "rent_check", "rent"}
            context_housing_inquiry = _is_housing_context_inquiry(req.message, latest_state)
            should_attach_houses = (
                bool(invoke_resp.candidates)
                or response_kind == "search"
                or intent_value in house_intents
                or context_housing_inquiry
            )
            if should_attach_houses:
                houses: list[str] = []
                seen_house_ids: set[str] = set()
                known_house_ids = _collect_house_ids_from_state(latest_state) | _collect_house_ids_from_tool_results(tool_results)
                for item in invoke_resp.candidates:
                    house_id = getattr(item, "house_id", None)
                    if isinstance(house_id, str) and house_id and house_id not in seen_house_ids:
                        seen_house_ids.add(house_id)
                        known_house_ids.add(house_id)
                        houses.append(house_id)
                for ref in invoke_resp.debug.get("referenced_house_ids", []):
                    if isinstance(ref, str) and ref and ref in known_house_ids and ref not in seen_house_ids:
                        seen_house_ids.add(ref)
                        houses.append(ref)
                if not houses and latest_state is not None and (
                    intent_value in {"rent_check", "rent", "house_detail", "listings", "amenities"} or context_housing_inquiry
                ):
                    focus_house_id = getattr(getattr(latest_state, "candidate_state", None), "focus_house_id", None)
                    if isinstance(focus_house_id, str) and focus_house_id and focus_house_id in known_house_ids:
                        houses.append(focus_house_id)
                if not houses and latest_state is not None and context_housing_inquiry:
                    latest_ids = getattr(getattr(latest_state, "candidate_state", None), "latest_house_ids", None)
                    if isinstance(latest_ids, list):
                        houses = [hid for hid in latest_ids if isinstance(hid, str) and hid][:5]
                response_payload: dict[str, Any] = {"message": output_text, "houses": houses}
            else:
                response_payload = {"message": output_text}

            timestamp = int(time.time())
            duration_ms = int((time.perf_counter() - started) * 1000)
            log_event(
                LOGGER,
                "chat.completed",
                step=STEP_FINAL_RESPONSE,
                duration_ms=duration_ms,
                invoke_output=preview_text(invoke_resp.text, limit=300),
                response=preview_payload(response_payload, limit=300),
            )
            return ChatResponse(
                session_id=req.session_id,
                response=response_payload,
                status="success",
                tool_results=tool_results,
                timestamp=timestamp,
                duration_ms=duration_ms,
            )
        finally:
            reset_tool_recording(tool_recorder_tokens)
            reset_log_context(tokens)

    @app.post("/vi/chat/completions")
    @app.post("/v1/chat/completions")
    async def proxy_chat_completions(
        payload: dict = Body(default_factory=dict),
        model_ip: str | None = Header(default=None, alias="Model-IP"),
        session_id: str | None = Header(default=None, alias="Session-ID"),
    ) -> dict:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session-ID header is required for /v1/chat/completions")

        resolved_model_ip = model_ip
        if not resolved_model_ip:
            resolved_model_ip = app.state.session_model_ip.get(session_id)
        if not resolved_model_ip:
            raise HTTPException(status_code=400, detail="model_ip is required (Model-IP header or known session)")

        http_client: httpx.AsyncClient = app.state.http_client
        headers: dict[str, str] = {}
        if session_id:
            headers["Session-ID"] = session_id

        response = await http_client.post(
            f"{_build_model_base_url(resolved_model_ip)}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    @app.get("/debug/agent-io/events")
    async def debug_agent_io_events(
        limit: int = 200,
        session_id: str | None = None,
        include_debug_endpoints: bool = False,
        compact: bool = False,
    ) -> dict[str, Any]:
        log_path = _resolve_agent_http_io_log_path()
        requested_limit = max(1, min(limit, 1000))
        scan_limit = max(200, requested_limit)
        if session_id:
            scan_limit = min(5000, max(scan_limit, requested_limit * 20))

        entries = _read_agent_http_io_entries(limit=scan_limit)
        if not include_debug_endpoints:
            entries = [item for item in entries if not _is_debug_agent_io_path(str(item.get("path", "")))]
        if session_id:
            entries = [item for item in entries if str(item.get("session_id", "")).strip() == session_id]
        entries = entries[-requested_limit:]

        enriched_raw = [{"stage": _stage_name(item), **item} for item in entries]
        if compact:
            enriched = [_compact_agent_io_entry_for_ui(item) for item in enriched_raw]
        else:
            enriched = [_normalize_agent_io_entry_for_display(item) for item in enriched_raw]
        return {
            "count": len(enriched),
            "items": enriched,
            "session_id": session_id or "",
            "log_path": str(log_path),
            "log_exists": log_path.exists(),
        }

    @app.get("/debug/agent-io", response_class=HTMLResponse)
    async def debug_agent_io_page() -> HTMLResponse:
        html = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent HTTP IO Monitor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; background: #f7f7f9; }
    h1 { margin: 0 0 12px; }
    .hint { color: #555; margin-bottom: 10px; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 8px; }
    .stage { font-weight: bold; color: #1f4b99; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 6px 0 0; }
  </style>
</head>
<body>
  <h1>Agent 交互实时面板</h1>
  <div class="hint">顺序关注：用户输入 → LLM输入 → LLM输出 → API调用 → API输出 → 最终回复。可在 URL 上附加 ?session_id=xxx 只看单会话。</div>
  <div id="list"></div>
  <script>
    var query = new URLSearchParams(window.location.search);
    var sessionId = query.get('session_id');

    function repeatPad(indent) {
      var pad = '';
      var i;
      for (i = 0; i < indent; i++) pad += '  ';
      return pad;
    }

    function formatValue(value, indent) {
      var i;
      var pad = repeatPad(indent || 0);
      if (value === null) return 'null';
      if (Object.prototype.toString.call(value) === '[object Array]') {
        if (value.length === 0) return '[]';
        var rows = [];
        for (i = 0; i < value.length; i++) {
          rows.push(pad + '  - ' + formatValue(value[i], (indent || 0) + 1));
        }
        return '[\\n' + rows.join('\\n') + '\\n' + pad + ']';
      }
      if (typeof value === 'object') {
        var keys = Object.keys(value || {});
        if (keys.length === 0) return '{}';
        var objRows = [];
        for (i = 0; i < keys.length; i++) {
          var k = keys[i];
          objRows.push(pad + '  ' + k + ': ' + formatValue(value[k], (indent || 0) + 1));
        }
        return '{\\n' + objRows.join('\\n') + '\\n' + pad + '}';
      }
      if (typeof value === 'string') {
        if (value.indexOf('\\n') === -1) return value;
        var lines = value.split('\\n');
        var textRows = [];
        for (i = 0; i < lines.length; i++) {
          textRows.push(pad + '  ' + lines[i]);
        }
        return '|\\n' + textRows.join('\\n');
      }
      return String(value);
    }

    function renderError(message) {
      var list = document.getElementById('list');
      list.innerHTML = '';
      var tip = document.createElement('div');
      tip.className = 'card';
      tip.textContent = message;
      list.appendChild(tip);
    }

    function load() {
      var params = new URLSearchParams({ limit: '200' });
      if (sessionId) params.set('session_id', sessionId);
      var xhr = new XMLHttpRequest();
      xhr.open('GET', '/debug/agent-io/events?' + params.toString(), true);
      xhr.onreadystatechange = function () {
        if (xhr.readyState !== 4) return;
        if (xhr.status < 200 || xhr.status >= 300) {
          renderError('日志加载失败：HTTP ' + xhr.status);
          return;
        }
        var data;
        try {
          data = JSON.parse(xhr.responseText || '{}');
        } catch (e) {
          renderError('日志加载失败：JSON解析错误');
          return;
        }

        var list = document.getElementById('list');
        list.innerHTML = '';

        var meta = document.createElement('div');
        meta.className = 'card';
        meta.textContent =
          'count=' + (data.count || 0) +
          ' | log_path=' + (data.log_path || '-') +
          ' | log_exists=' + (data.log_exists ? 'yes' : 'no');
        list.appendChild(meta);

        if (!data.items || !data.items.length) {
          var emptyTip = document.createElement('div');
          emptyTip.className = 'card';
          emptyTip.textContent = '暂无日志。可先发一条 /api/v1/chat 请求，或检查 session_id 过滤、日志路径。';
          list.appendChild(emptyTip);
          return;
        }

        var j;
        for (j = 0; j < data.items.length; j++) {
          var item = data.items[j];
          var div = document.createElement('div');
          div.className = 'card';
          var stage = document.createElement('div');
          stage.className = 'stage';
          stage.textContent = item.stage || '其他';
          var pre = document.createElement('pre');
          pre.textContent = formatValue(item, 0);
          div.appendChild(stage);
          div.appendChild(pre);
          list.appendChild(div);
        }
      };
      xhr.send(null);
    }

    load();
    setInterval(load, 4000);
  </script>
</body>
</html>
        """
        return HTMLResponse(content=html)

    @app.post("/v2/chat/completions")
    async def proxy_chat_completions_v2(
        payload: dict = Body(default_factory=dict),
        model_ip: str | None = Header(default=None, alias="Model-IP"),
    ) -> dict:
        if not model_ip:
            raise HTTPException(status_code=400, detail="Model-IP header is required for /v2/chat/completions")

        http_client: httpx.AsyncClient = app.state.http_client
        response = await http_client.post(
            f"{_build_model_base_url(model_ip)}/v1/chat/completions",
            json=payload,
            headers={},
        )
        response.raise_for_status()
        return response.json()

    return app


app = create_app()
