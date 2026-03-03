from __future__ import annotations

import contextvars
import json
import logging
import os
from typing import Any

_CTX_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_CTX_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
_CTX_CASE_TYPE: contextvars.ContextVar[str] = contextvars.ContextVar("case_type", default="-")
_CTX_USER_ID: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")
_CTX_SEQ: contextvars.ContextVar[int] = contextvars.ContextVar("log_seq", default=0)
_HTTP_IO_LOGGER_NAME = "agent.http.io"
_HTTP_IO_LOG_PATH = "agent_http_io.log"


def setup_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
    else:
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    _ensure_http_io_logger()


def _ensure_http_io_logger() -> None:
    logger = logging.getLogger(_HTTP_IO_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = os.getenv("AGENT_HTTP_IO_LOG_PATH", _HTTP_IO_LOG_PATH)
    abs_log_path = os.path.abspath(log_path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_log_path:
            return

    file_handler = logging.FileHandler(abs_log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(file_handler)


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


def get_log_context() -> dict[str, str]:
    return {
        "trace_id": _CTX_TRACE_ID.get(),
        "session_id": _CTX_SESSION_ID.get(),
        "case_type": _CTX_CASE_TYPE.get(),
        "user_id": _CTX_USER_ID.get(),
    }


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


def log_json_event(logger: logging.Logger, payload: dict[str, Any]) -> None:
    try:
        message = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        message = str(payload)
    logger.info("%s", message)


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
    logger.info("%s", preview_payload(payload, limit=4000))
