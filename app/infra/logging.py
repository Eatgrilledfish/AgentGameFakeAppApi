from __future__ import annotations

import contextvars
import json
import logging
from collections import defaultdict, deque
from threading import Lock
from typing import Any

_CTX_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_CTX_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
_CTX_CASE_TYPE: contextvars.ContextVar[str] = contextvars.ContextVar("case_type", default="-")
_CTX_USER_ID: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")
_CTX_SEQ: contextvars.ContextVar[int] = contextvars.ContextVar("log_seq", default=0)

_TRACE_LIMIT = 300
_ALL_TRACE_LIMIT = 3000
_TRACE_STORE: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=_TRACE_LIMIT))
_ALL_TRACE_STORE: deque[dict[str, Any]] = deque(maxlen=_ALL_TRACE_LIMIT)
_TRACE_LOCK = Lock()
_TRACE_EVENT_ID = 0


def setup_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        return
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def bind_log_context(
    *,
    trace_id: str,
    session_id: str,
    case_type: str,
    user_id: str,
) -> tuple[contextvars.Token[str], contextvars.Token[str], contextvars.Token[str], contextvars.Token[str], contextvars.Token[int]]:
    t1 = _CTX_TRACE_ID.set(trace_id)
    t2 = _CTX_SESSION_ID.set(session_id)
    t3 = _CTX_CASE_TYPE.set(case_type)
    t4 = _CTX_USER_ID.set(user_id)
    t5 = _CTX_SEQ.set(0)
    return (t1, t2, t3, t4, t5)


def reset_log_context(
    tokens: tuple[contextvars.Token[str], contextvars.Token[str], contextvars.Token[str], contextvars.Token[str], contextvars.Token[int]]
) -> None:
    t1, t2, t3, t4, t5 = tokens
    _CTX_TRACE_ID.reset(t1)
    _CTX_SESSION_ID.reset(t2)
    _CTX_CASE_TYPE.reset(t3)
    _CTX_USER_ID.reset(t4)
    _CTX_SEQ.reset(t5)


def preview_payload(payload: Any, limit: int = 1000) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        text = str(payload)
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def preview_text(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    seq = _CTX_SEQ.get() + 1
    _CTX_SEQ.set(seq)
    payload: dict[str, Any] = {
        "seq": seq,
        "trace_id": _CTX_TRACE_ID.get(),
        "session_id": _CTX_SESSION_ID.get(),
        "case_type": _CTX_CASE_TYPE.get(),
        "user_id": _CTX_USER_ID.get(),
        "event": event,
    }
    payload.update(fields)
    _append_trace_event(payload)
    logger.info("%s", preview_payload(payload, limit=4000))


def _append_trace_event(payload: dict[str, Any]) -> None:
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id or session_id == "-":
        return
    global _TRACE_EVENT_ID
    with _TRACE_LOCK:
        _TRACE_EVENT_ID += 1
        event = payload.copy()
        event["eid"] = _TRACE_EVENT_ID
        _TRACE_STORE[session_id].append(event)
        _ALL_TRACE_STORE.append(event)


def get_trace_events(session_id: str) -> list[dict[str, Any]]:
    with _TRACE_LOCK:
        return list(_TRACE_STORE.get(session_id, []))


def get_all_trace_events(*, since_eid: int = 0) -> list[dict[str, Any]]:
    with _TRACE_LOCK:
        if since_eid <= 0:
            return list(_ALL_TRACE_STORE)
        return [event for event in _ALL_TRACE_STORE if int(event.get("eid", 0)) > since_eid]


def clear_trace_events(session_id: str) -> int:
    with _TRACE_LOCK:
        existing = _TRACE_STORE.get(session_id)
        if not existing:
            return 0
        count = len(existing)
        existing.clear()
        return count


def clear_all_trace_events() -> int:
    with _TRACE_LOCK:
        total = len(_ALL_TRACE_STORE)
        _ALL_TRACE_STORE.clear()
        _TRACE_STORE.clear()
        return total
