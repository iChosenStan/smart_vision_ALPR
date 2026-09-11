"""Serviço que agrega métricas para o dashboard principal."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.ticket import TicketStatus
from backend.app.repositories.ticket_repository import TicketRepository
from backend.app.repositories.vehicle_repository import VehicleRepository


@dataclass
class DashboardSummary:
    total_vehicles: int
    vehicles_parked: int
    tickets_paid: int
    tickets_pending: int
    simulated_revenue: float
    available_spots: int
    total_spots: int


class DashboardService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.ticket_repo = TicketRepository(db)
        self.vehicle_repo = VehicleRepository(db)

    def get_summary(self) -> DashboardSummary:
        vehicles_parked = self.ticket_repo.count_parked()
        total_spots = settings.total_parking_spots

        return DashboardSummary(
            total_vehicles=self.vehicle_repo.count_all(),
            vehicles_parked=vehicles_parked,
            tickets_paid=self.ticket_repo.count_by_status(TicketStatus.PAGO),
            tickets_pending=self.ticket_repo.count_by_status(TicketStatus.EM_ABERTO),
            simulated_revenue=self.ticket_repo.sum_revenue(),
            available_spots=max(0, total_spots - vehicles_parked),
            total_spots=total_spots,
        )
