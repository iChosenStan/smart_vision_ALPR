"""Repository de Payment."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from backend.app.models.payment import Payment


class PaymentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, ticket_id: int, amount: float, method: str, paid_at: datetime) -> Payment:
        payment = Payment(ticket_id=ticket_id, amount=amount, method=method, paid_at=paid_at)
        self.db.add(payment)
        self.db.flush()
        return payment
