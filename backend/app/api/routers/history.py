"""Endpoint de histórico — listagem completa de tickets, com filtros
(placa, status, data, e implicitamente tempo de permanência/valor pago,
já expostos em cada item via `TicketRead`)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.ticket import TicketStatus
from backend.app.schemas.ticket import TicketListResponse
from backend.app.services.ticket_query_service import TicketQueryService

router = APIRouter(tags=["history"])


@router.get("/history", response_model=TicketListResponse)
def get_history(
    plate: Optional[str] = None,
    status: Optional[TicketStatus] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
) -> TicketListResponse:
    return TicketQueryService(db).list_tickets(
        plate=plate, status=status, date_from=date_from, date_to=date_to, page=page, page_size=page_size
    )
