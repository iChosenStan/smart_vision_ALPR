"""DTOs Pydantic do dashboard."""

from __future__ import annotations

from pydantic import BaseModel


class DashboardResponse(BaseModel):
    total_vehicles: int
    vehicles_parked: int
    tickets_paid: int
    tickets_pending: int
    simulated_revenue: float
    available_spots: int
    total_spots: int
