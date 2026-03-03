from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import Any
from uuid import uuid4

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from app.agent.service import AgentService
from app.agent.state import StateStore
from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.infra.cache import CacheManager
from app.infra.logging import bind_log_context, get_log_context, log_event, preview_payload, preview_text, reset_log_context, setup_logging
from app.infra.tool_recorder import begin_tool_recording, get_tool_results, reset_tool_recording
from app.schemas import CaseType, ChatRequest, ChatResponse, HealthResponse, InvokeRequest, InvokeResponse
from app.settings import AgentSettings, load_settings

LOGGER = logging.getLogger(__name__)
HTTP_IO_LOGGER = logging.getLogger("agent.http.io")

STEP_RECV_USER_QUERY = "STEP-01-RECV-USER-QUERY"
STEP_AGENT_PIPELINE = "STEP-02-AGENT-PIPELINE"
STEP_LLM_FALLBACK = "STEP-03-LLM-FALLBACK"
STEP_LLM_NLU = "STEP-02A-LLM-NLU"
STEP_FINAL_RESPONSE = "STEP-04-FINAL-RESPONSE"
STEP_HTTP = "STEP-00-HTTP"

LLM_TIMEOUT = httpx.Timeout(connect=1.0, read=6.0, write=2.0, pool=0.5)
LLM_RETRYABLE_EXCEPTIONS = (httpx.ConnectTimeout, httpx.ReadTimeout)


def _preview_http_body(raw_body: bytes, content_type: str | None, limit: int = 2000) -> str:
    if not raw_body:
        return "<empty>"

    text = raw_body.decode("utf-8", errors="replace")
    if content_type and "application/json" in content_type.lower():
        parsed = _extract_json_object(text)
        if parsed is not None:
            return preview_payload(parsed, limit=limit)
    return preview_text(text, limit=limit)


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

def _read_agent_http_io_entries(limit: int = 200) -> list[dict[str, Any]]:
    log_path = os.getenv("AGENT_HTTP_IO_LOG_PATH", "agent_http_io.log")
    path = Path(log_path)
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
    session_id: str | None = None,
) -> str | None:
    headers: dict[str, str] = {}
    if session_id:
        headers["Session-ID"] = session_id

    payload = {
        "model": "",
        "messages": messages,
        "max_tokens": 256,
        "stream": False,
    }
    target_url = f"{_build_model_base_url(model_ip)}/v1/chat/completions"
    started = time.perf_counter()
    log_event(
        LOGGER,
        "chat.llm.forward.request",
        step=STEP_LLM_FALLBACK,
        model_ip=model_ip,
        payload_preview=preview_payload(payload),
    )
    HTTP_IO_LOGGER.info(
        "%s",
        preview_payload(
            {
                **get_log_context(),
                "event": "http.agent_io.llm.request",
                "method": "POST",
                "url": target_url,
                "session_id": session_id or "-",
                "request_content_type": "application/json",
                "request_body": preview_payload(payload, limit=8000),
            },
            limit=8000,
        ),
    )

    data: dict[str, Any]
    for attempt in range(2):
        try:
            resp = await _llm_post(
                http_client,
                url=target_url,
                payload=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except LLM_RETRYABLE_EXCEPTIONS as exc:
            if attempt == 0:
                await asyncio.sleep(random.uniform(0.05, 0.15))
                continue
            HTTP_IO_LOGGER.info(
                "%s",
                preview_payload(
                    {
                        **get_log_context(),
                        "event": "http.agent_io.llm.error",
                        "method": "POST",
                        "url": target_url,
                        "session_id": session_id or "-",
                        **_normalize_error_payload(exc),
                        "attempt": attempt + 1,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                    },
                    limit=8000,
                ),
            )
            raise
        except Exception as exc:
            HTTP_IO_LOGGER.info(
                "%s",
                preview_payload(
                    {
                        **get_log_context(),
                        "event": "http.agent_io.llm.error",
                        "method": "POST",
                        "url": target_url,
                        "session_id": session_id or "-",
                        **_normalize_error_payload(exc),
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                    },
                    limit=8000,
                ),
            )
            raise

    log_event(
        LOGGER,
        "chat.llm.forward.response",
        step=STEP_LLM_FALLBACK,
        status_code=resp.status_code,
        body_preview=preview_payload(data),
    )
    HTTP_IO_LOGGER.info(
        "%s",
        preview_payload(
            {
                **get_log_context(),
                "event": "http.agent_io.llm.response",
                "method": "POST",
                "url": target_url,
                "session_id": session_id or "-",
                "status_code": resp.status_code,
                "response_content_type": getattr(resp, "headers", {}).get("content-type", "application/json"),
                "response_body": preview_payload(data, limit=8000),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
            limit=8000,
        ),
    )

    choices = data.get("choices", [])
    if not choices:
        return None
    msg = choices[0].get("message", {})
    return msg.get("content")


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


def _sanitize_llm_parse(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parsed)
    tool_plan = normalized.get("tool_plan")
    if not isinstance(tool_plan, dict):
        return normalized

    operation_id = tool_plan.get("operationId")
    arguments = tool_plan.get("arguments")
    if not isinstance(operation_id, str) or not operation_id:
        return normalized

    known_op_specs = _load_tool_argument_specs()
    if operation_id != "none" and operation_id not in known_op_specs:
        normalized["tool_plan"] = {"operationId": "none", "arguments": {}}
        return normalized

    sanitized_arguments: dict[str, Any] = {}
    op_specs = known_op_specs.get(operation_id, {})
    if isinstance(arguments, dict):
        if op_specs:
            for key, value in arguments.items():
                if not isinstance(key, str):
                    continue
                spec = op_specs.get(key)
                if spec is None:
                    continue
                coerced = _coerce_argument_value(value, spec)
                if coerced is not None:
                    sanitized_arguments[key] = coerced
        else:
            sanitized_arguments = {k: v for k, v in arguments.items() if isinstance(k, str)}

    required = _load_tool_required_params().get(operation_id, set())
    for req_key in required:
        if req_key not in sanitized_arguments:
            continue
        if sanitized_arguments[req_key] is None:
            sanitized_arguments.pop(req_key, None)

    normalized["tool_plan"] = {
        "operationId": operation_id,
        "arguments": sanitized_arguments,
    }
    return normalized


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
        try:
            return int(float(stripped))
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


def _build_llm_context_facts(state: Any) -> dict[str, Any]:
    if state is None:
        return {}

    facts: dict[str, Any] = {}

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

    search_history = getattr(state, "search_history", None)
    if isinstance(search_history, list) and search_history:
        recent_searches: list[dict[str, Any]] = []
        for snap in search_history[-3:]:
            if not hasattr(snap, "house_ids"):
                continue
            house_ids = getattr(snap, "house_ids", [])
            if not isinstance(house_ids, list) or not house_ids:
                continue
            row: dict[str, Any] = {"house_ids": [str(x) for x in house_ids[:3]]}
            for key in ("district", "area", "community", "landmark_name"):
                value = getattr(snap, key, None)
                if isinstance(value, str) and value:
                    row[key] = value
            recent_searches.append(row)
        if recent_searches:
            facts["recent_searches"] = recent_searches
            facts["latest_search_house_ids"] = recent_searches[-1]["house_ids"]

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


def _build_llm_nlu_messages(message: str, summary: str, context_facts: dict[str, Any] | None = None) -> list[dict[str, str]]:
    tool_summary = _load_tool_schema_summary()
    prompt = (
        "你是租房Agent意图解析器。"
        "请把输入解析为严格JSON，不要输出任何额外文字或Markdown。\n"
        "intent可选值：chat/search/compare/amenities/house_detail/listings/rent/terminate/offline。\n"
        "你必须基于可用API工具目录判断本轮最匹配的operationId，并抽取调用参数。\n"
        "严格反幻觉：不允许编造 house_id、listing_platform、地标ID 或未提供的硬约束。\n"
        "若用户文本里显式出现 house_id（如 HF_67），必须使用该 house_id，不得被上下文覆盖。\n"
        "仅当用户未显式给出 house_id 且出现“这套/第一套/最开始那套”时，才可使用 context_facts 中的 focus_house_id 或 latest_search_house_ids。\n"
        "若必填参数未知，不要猜测，保留为空并降低confidence。\n"
        "只返回如下结构："
        '{"intent":"search","tool_plan":{"operationId":"get_houses_by_platform","arguments":{}},"hard":{},"soft":{},"confidence":0.0}。\n'
        "operationId必须从工具目录中选择；若是纯闲聊可填 none。\n"
        "hard可含字段：district,area,community,landmark_id,landmark_name,landmark_category,budget_min,budget_max,rent_type,"
        "layout,area_min,max_subway_dist,max_commute_min,utilities_type,move_in_date,listing_platform,house_id。\n"
        "soft可含字段：decoration,elevator,orientation,noise_preference,amenities,value_for_money,prioritize_subway_distance。\n"
        "tool_plan.arguments请使用API参数名（如 max_price/rental_type/listing_platform/house_id）。\n"
        "tool_plan.arguments必须遵守argdefs中的参数类型(type)和必填(required)，不要返回未定义参数。\n"
        "未提及字段不要猜测，可以不返回或设为null。\n"
        f"工具目录:\n{tool_summary}"
    )
    user_payload = {
        "history_summary": summary[:500] if summary else "",
        "context_facts": context_facts or {},
        "message": message,
    }
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


async def _analyze_intent_with_llm(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    session_id: str,
    message: str,
    summary: str,
    context_facts: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    llm_text = await _forward_chat_completion(
        http_client,
        model_ip=model_ip,
        messages=_build_llm_nlu_messages(message, summary, context_facts=context_facts),
        session_id=session_id,
    )
    if not llm_text:
        return None
    parsed = _extract_json_object(llm_text)
    if not parsed:
        return None
    return _sanitize_llm_parse(parsed)


def create_app(settings: AgentSettings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    setup_logging(cfg.log_level)
    # Fail fast at startup if compact LLM tool schema is unavailable.
    _load_tool_schema_summary()

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
            HTTP_IO_LOGGER.info(
                "%s",
                preview_payload(
                    {
                        **get_log_context(),
                        "event": "http.agent_io.request",
                        "session_id": http_session_id,
                        "method": request.method,
                        "path": request.url.path,
                        "query": str(request.query_params),
                        "request_content_type": req_content_type or "<unknown>",
                        "request_body": _preview_http_body(raw_body, req_content_type),
                    },
                    limit=8000,
                ),
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
            HTTP_IO_LOGGER.info(
                "%s",
                preview_payload(
                    {
                        **get_log_context(),
                        "event": "http.agent_io.response",
                        "session_id": http_session_id,
                        "method": request.method,
                        "path": request.url.path,
                        "query": str(request.query_params),
                        "status_code": replay_response.status_code,
                        "response_content_type": resp_content_type or "<unknown>",
                        "response_body": _preview_http_body(response_body, resp_content_type),
                        "duration_ms": duration_ms,
                    },
                    limit=8000,
                ),
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
        tokens = bind_log_context(
            trace_id=trace_id,
            session_id=req.session_id,
            case_type=CaseType.single.value,
            user_id=cfg.default_user_id,
        )
        tool_recorder_tokens = begin_tool_recording()
        service: AgentService = app.state.agent_service
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
            if hasattr(service, "state_store"):
                state = service.state_store.get(req.session_id)  # type: ignore[attr-defined]
                if state is not None and getattr(state, "conversation_summary", ""):
                    state_summary = state.conversation_summary
                    context_facts = _build_llm_context_facts(state)

            prompt_basis = req.message + state_summary + json.dumps(context_facts, ensure_ascii=False)
            nlu_prompt_tokens = (
                service.rough_token_estimate(prompt_basis)
                if hasattr(service, "rough_token_estimate")
                else max(1, int(len(prompt_basis) / 2))
            )
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

            log_event(
                LOGGER,
                "chat.agent.invoke.start",
                step=STEP_AGENT_PIPELINE,
                input_message=preview_text(req.message, limit=300),
            )
            invoke_resp = await service.handle(
                InvokeRequest(
                    session_id=req.session_id,
                    case_type=CaseType.single,
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
            if response_kind == "search" or bool(invoke_resp.candidates):
                response_payload: dict[str, Any] = {
                    "message": invoke_resp.text,
                    "houses": [item.house_id for item in invoke_resp.candidates],
                }
            elif response_kind == "chat":
                try:
                    llm_messages = (
                        service.build_llm_fallback_messages(req.session_id, req.message)
                        if hasattr(service, "build_llm_fallback_messages")
                        else [{"role": "user", "content": req.message}]
                    )
                    llm_prompt_tokens = (
                        service.rough_token_estimate(json.dumps(llm_messages, ensure_ascii=False))
                        if hasattr(service, "rough_token_estimate")
                        else max(1, int(len(req.message) / 2))
                    )
                    http_client: httpx.AsyncClient = app.state.http_client
                    llm_text = await _forward_chat_completion(
                        http_client,
                        model_ip=req.model_ip,
                        messages=llm_messages,
                        session_id=req.session_id,
                    )
                    if llm_text:
                        output_text = llm_text
                        completion_tokens = (
                            service.rough_token_estimate(llm_text)
                            if hasattr(service, "rough_token_estimate")
                            else max(1, int(len(llm_text) / 2))
                        )
                        if hasattr(service, "record_llm_fallback_usage"):
                            service.record_llm_fallback_usage(req.session_id, llm_prompt_tokens + completion_tokens)
                        log_event(
                            LOGGER,
                            "chat.llm_fallback.applied",
                            step=STEP_LLM_FALLBACK,
                            prompt_tokens=llm_prompt_tokens,
                            completion_tokens=completion_tokens,
                            llm_output=preview_text(llm_text, limit=300),
                        )
                except httpx.HTTPError as exc:
                    log_event(
                        LOGGER,
                        "chat.llm_fallback_failed",
                        step=STEP_LLM_FALLBACK,
                        error=str(exc),
                    )
                response_payload = {"message": output_text}
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
                tool_results=get_tool_results(),
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
    ) -> dict[str, Any]:
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

        enriched = [{"stage": _stage_name(item), **item} for item in entries]
        return {
            "count": len(enriched),
            "items": enriched,
            "session_id": session_id or "",
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
    const query = new URLSearchParams(window.location.search);
    const sessionId = query.get('session_id');

    async function load() {
      const params = new URLSearchParams({ limit: '200' });
      if (sessionId) params.set('session_id', sessionId);
      const res = await fetch('/debug/agent-io/events?' + params.toString());
      const data = await res.json();
      const list = document.getElementById('list');
      list.innerHTML = '';
      for (const item of data.items) {
        const div = document.createElement('div');
        div.className = 'card';
        const stage = document.createElement('div');
        stage.className = 'stage';
        stage.textContent = item.stage || '其他';
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(item, null, 2);
        div.appendChild(stage);
        div.appendChild(pre);
        list.appendChild(div);
      }
    }
    load();
    setInterval(load, 2000);
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
