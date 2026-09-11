"""Repository de Ticket."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from backend.app.models.ticket import Ticket, TicketStatus
from backend.app.models.vehicle import Vehicle


class TicketRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, **kwargs: Any) -> Ticket:
        ticket = Ticket(**kwargs)
        self.db.add(ticket)
        self.db.flush()
        return ticket

    def get_by_id(self, ticket_id: int) -> Optional[Ticket]:
        """Busca já carregando o relacionamento `vehicle` (evita erro de
        sessão fechada ao serializar a resposta HTTP)."""
        return (
            self.db.query(Ticket)
            .options(joinedload(Ticket.vehicle))
            .filter(Ticket.id == ticket_id)
            .first()
        )

    def get_open_ticket_by_vehicle(self, vehicle_id: int) -> Optional[Ticket]:
        """Retorna o ticket mais recente do veículo que ainda não foi finalizado."""
        return (
            self.db.query(Ticket)
            .filter(Ticket.vehicle_id == vehicle_id, Ticket.status != TicketStatus.FINALIZADO)
            .order_by(Ticket.entry_at.desc())
            .first()
        )

    def list_filtered(
        self,
        plate: Optional[str] = None,
        status: Optional[TicketStatus] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Ticket], int]:
        """Lista tickets com filtros opcionais, paginado. Usado tanto por
        `GET /api/tickets` (tabela em tempo real) quanto por
        `GET /api/history` (histórico completo)."""
        query = self.db.query(Ticket).options(joinedload(Ticket.vehicle)).join(Vehicle)

        if plate:
            query = query.filter(Vehicle.plate.ilike(f"%{plate}%"))
        if status:
            query = query.filter(Ticket.status == status)
        if date_from:
            query = query.filter(Ticket.entry_at >= date_from)
        if date_to:
            query = query.filter(Ticket.entry_at <= date_to)

        total = query.count()
        items = (
            query.order_by(Ticket.entry_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return items, total

    def count_by_status(self, status: TicketStatus) -> int:
        return self.db.query(Ticket).filter(Ticket.status == status).count()

    def count_parked(self) -> int:
        """Veículos atualmente no estacionamento: entrada registrada e
        ainda não finalizada (independente de já estar pago ou não)."""
        return (
            self.db.query(Ticket)
            .filter(Ticket.status.in_([TicketStatus.EM_ABERTO, TicketStatus.PAGO]))
            .count()
        )

    def sum_revenue(self) -> float:
        """Soma do valor de todos os tickets pagos (PAGO ou já FINALIZADO)."""
        result = (
            self.db.query(func.coalesce(func.sum(Ticket.amount), 0.0))
            .filter(Ticket.status.in_([TicketStatus.PAGO, TicketStatus.FINALIZADO]))
            .scalar()
        )
        return float(result or 0.0)
