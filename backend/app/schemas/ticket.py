"""DTOs Pydantic relacionados a Ticket."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, computed_field

from backend.app.models.ticket import TicketStatus
from backend.app.schemas.vehicle import VehicleRead


class TicketRead(BaseModel):
    """Representa o "ticket virtual" exibido no frontend (Etapa 2 do
    fluxo de entrada, conforme especificação)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    vehicle: VehicleRead
    entry_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None
    entry_at: datetime
    exit_at: Optional[datetime] = None
    ocr_confidence: float
    status: TicketStatus
    amount: Optional[float] = None
    payment_method: Optional[str] = None
    paid_at: Optional[datetime] = None

    @computed_field
    @property
    def ticket_number(self) -> str:
        return f"T{self.id:06d}"

    @computed_field
    @property
    def duration_minutes(self) -> int:
        """Tempo de permanência em minutos — até `exit_at`, ou até agora
        se o veículo ainda estiver no estacionamento."""
        end = self.exit_at or datetime.now(timezone.utc)
        entry = self.entry_at if self.entry_at.tzinfo else self.entry_at.replace(tzinfo=timezone.utc)
        end = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        return max(0, int((end - entry).total_seconds() // 60))


class TicketListResponse(BaseModel):
    """Resposta paginada de GET /api/tickets e GET /api/history."""

    items: List[TicketRead]
    total: int
    page: int
    page_size: int

