"""Repository de Event — registra a trilha de auditoria do sistema."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.app.models.event import Event, EventType


class EventRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def log(
        self, type_: EventType, ticket_id: Optional[int] = None, payload: Optional[Dict[str, Any]] = None
    ) -> Event:
        event = Event(type=type_, ticket_id=ticket_id, payload_json=payload)
        self.db.add(event)
        self.db.flush()
        return event
