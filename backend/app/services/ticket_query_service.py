"""Serviço de consulta/listagem de tickets — reaproveitado por
`GET /api/tickets` (tabela em tempo real) e `GET /api/history`
(histórico completo), evitando duplicar a lógica de filtro/paginação.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.models.ticket import TicketStatus
from backend.app.repositories.ticket_repository import TicketRepository
from backend.app.schemas.ticket import TicketListResponse, TicketRead


class TicketQueryService:
    def __init__(self, db: Session) -> None:
        self.ticket_repo = TicketRepository(db)

    def list_tickets(
        self,
        plate: Optional[str] = None,
        status: Optional[TicketStatus] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> TicketListResponse:
        items, total = self.ticket_repo.list_filtered(
            plate=plate,
            status=status,
            date_from=date_from,
            date_to=date_to,
            page=page,
            page_size=page_size,
        )
        return TicketListResponse(
            items=[TicketRead.model_validate(t) for t in items],
            total=total,
            page=page,
            page_size=page_size,
        )
