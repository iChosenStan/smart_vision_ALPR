"""Endpoints relacionados a Ticket: listagem, consulta e pagamento."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.ticket import TicketStatus
from backend.app.repositories.ticket_repository import TicketRepository
from backend.app.schemas.payment import PaymentRequest
from backend.app.schemas.ticket import TicketListResponse, TicketRead
from backend.app.services.payment_service import PaymentService, TicketAlreadyPaidError, TicketNotFoundError
from backend.app.services.ticket_query_service import TicketQueryService

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.get("", response_model=TicketListResponse)
def list_tickets(
    plate: Optional[str] = None,
    status: Optional[TicketStatus] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
) -> TicketListResponse:
    """Tabela em tempo real da tela principal — mesmos filtros do histórico."""
    return TicketQueryService(db).list_tickets(
        plate=plate, status=status, date_from=date_from, date_to=date_to, page=page, page_size=page_size
    )


@router.get("/{ticket_id}", response_model=TicketRead)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)) -> TicketRead:
    ticket = TicketRepository(db).get_by_id(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket #{ticket_id} não encontrado.")
    return TicketRead.model_validate(ticket)


@router.put("/{ticket_id}/pay", response_model=TicketRead)
def pay_ticket(ticket_id: int, payload: PaymentRequest, db: Session = Depends(get_db)) -> TicketRead:
    service = PaymentService(db)
    try:
        ticket = service.pay_ticket(ticket_id, payload.payment_method)
    except TicketNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TicketAlreadyPaidError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return TicketRead.model_validate(ticket)
