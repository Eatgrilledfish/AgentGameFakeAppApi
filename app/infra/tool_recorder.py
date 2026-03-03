from __future__ import annotations

import contextvars
from typing import Any

_CTX_TOOL_RESULTS: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "tool_results",
    default=None,
)
_CTX_TOOL_SEQ: contextvars.ContextVar[int] = contextvars.ContextVar("tool_seq", default=0)


def begin_tool_recording() -> tuple[contextvars.Token[list[dict[str, Any]] | None], contextvars.Token[int]]:
    rows_token = _CTX_TOOL_RESULTS.set([])
    seq_token = _CTX_TOOL_SEQ.set(0)
    return rows_token, seq_token


def reset_tool_recording(
    tokens: tuple[contextvars.Token[list[dict[str, Any]] | None], contextvars.Token[int]]
) -> None:
    rows_token, seq_token = tokens
    _CTX_TOOL_RESULTS.reset(rows_token)
    _CTX_TOOL_SEQ.reset(seq_token)


def get_tool_results() -> list[dict[str, Any]]:
    rows = _CTX_TOOL_RESULTS.get()
    if not rows:
        return []
    return [dict(item) for item in rows]


def record_tool_result(
    *,
    name: str,
    success: bool,
    output: Any,
    duration_ms: int,
    method: str,
    url: str,
    status_code: int | None = None,
) -> None:
    rows = _CTX_TOOL_RESULTS.get()
    if rows is None:
        return

    seq = _CTX_TOOL_SEQ.get() + 1
    _CTX_TOOL_SEQ.set(seq)

    rows.append(
        {
            "seq": seq,
            "name": name,
            "success": success,
            "output": _normalize_output(output),
            "duration_ms": max(0, duration_ms),
            "method": method,
            "url": url,
            "status_code": status_code,
        }
    )


def _normalize_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if isinstance(output, list):
        return {"items": output}
    if output is None:
        return {}
    return {"value": output}
