"""Modelo ORM: Ticket — o coração do fluxo de estacionamento."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


class TicketStatus(str, enum.Enum):
    EM_ABERTO = "EM_ABERTO"
    PAGO = "PAGO"
    FINALIZADO = "FINALIZADO"


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    vehicle_id: Mapped[int] = mapped_column(ForeignKey("vehicles.id"), nullable=False)

    entry_image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    plate_image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    entry_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    exit_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[TicketStatus] = mapped_column(Enum(TicketStatus), default=TicketStatus.EM_ABERTO)

    amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    payment_method: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    paid_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    vehicle: Mapped["Vehicle"] = relationship(back_populates="tickets")  # noqa: F821
    payments: Mapped[List["Payment"]] = relationship(back_populates="ticket")  # noqa: F821
    events: Mapped[List["Event"]] = relationship(back_populates="ticket")  # noqa: F821
