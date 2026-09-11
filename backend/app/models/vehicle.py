"""Modelo ORM: Vehicle — identificado unicamente pela placa."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(primary_key=True)
    plate: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    make: Mapped[str] = mapped_column(String(50), default="")
    model: Mapped[str] = mapped_column(String(50), default="")
    color: Mapped[str] = mapped_column(String(30), default="")
    type: Mapped[str] = mapped_column(String(30), default="")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    tickets: Mapped[List["Ticket"]] = relationship(back_populates="vehicle")  # noqa: F821
