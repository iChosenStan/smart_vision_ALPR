"""Modelo ORM: Event — trilha de auditoria (entrada, saída, pagamento, negado)."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


class EventType(str, enum.Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    PAYMENT = "PAYMENT"
    DENIED = "DENIED"


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[Optional[int]] = mapped_column(ForeignKey("tickets.id"), nullable=True)
    type: Mapped[EventType] = mapped_column(Enum(EventType), nullable=False)
    payload_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    ticket: Mapped[Optional["Ticket"]] = relationship(back_populates="events")  # noqa: F821
