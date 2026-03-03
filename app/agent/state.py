from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.schemas import CaseType, SessionState
from app.settings import AgentSettings


@dataclass(slots=True)
class _StateRecord:
    state: SessionState
    expires_at: datetime


class StateStore:
    def __init__(self, settings: AgentSettings) -> None:
        self._ttl = timedelta(seconds=settings.state_ttl_sec)
        self._items: dict[str, _StateRecord] = {}

    def _cleanup(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [sid for sid, record in self._items.items() if record.expires_at < now]
        for sid in expired:
            self._items.pop(sid, None)

    def get(self, session_id: str) -> SessionState | None:
        self._cleanup()
        record = self._items.get(session_id)
        if not record:
            return None
        record.expires_at = datetime.now(timezone.utc) + self._ttl
        record.state.updated_at = datetime.now(timezone.utc)
        return record.state

    def upsert(self, state: SessionState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        self._items[state.session_id] = _StateRecord(
            state=state,
            expires_at=datetime.now(timezone.utc) + self._ttl,
        )

    def get_or_create(self, session_id: str, user_id: str, case_type: CaseType) -> tuple[SessionState, bool]:
        existing = self.get(session_id)
        if existing:
            if existing.case_type != case_type:
                existing.case_type = case_type
                existing.updated_at = datetime.now(timezone.utc)
                self.upsert(existing)
            return existing, False
        state = SessionState(session_id=session_id, user_id=user_id, case_type=case_type)
        self.upsert(state)
        return state, True
