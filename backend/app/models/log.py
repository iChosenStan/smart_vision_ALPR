"""Modelo ORM: Log — registro bruto de cada leitura de OCR, mesmo as que
não resultam em ticket (ex: erro de OCR, veículo não identificado)."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Enum, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class LogSource(str, enum.Enum):
    ENTRY_CAM = "entry_cam"
    EXIT_CAM = "exit_cam"


class Log(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[LogSource] = mapped_column(Enum(LogSource), nullable=False)
    raw_ocr_text: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
