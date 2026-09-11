"""Serviço de simulação de pagamento de um ticket.

DECISÃO DE NEGÓCIO: o valor é calculado no momento do pagamento (não da
saída), cobrando por hora cheia (arredondada para cima), com mínimo de 1
hora — prática comum em estacionamentos reais. Como o pagamento pode
acontecer antes da saída física (conforme o fluxo especificado: botão
"Pagar" no dashboard, independente de quando o veículo efetivamente sai),
o tempo de permanência final pode diferir ligeiramente do cobrado — é uma
simplificação aceitável para o PoC.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.models.event import EventType
from backend.app.models.ticket import Ticket, TicketStatus
from backend.app.repositories.event_repository import EventRepository
from backend.app.repositories.payment_repository import PaymentRepository
from backend.app.repositories.ticket_repository import TicketRepository

logger = get_logger(__name__)


class TicketNotFoundError(Exception):
    pass


class TicketAlreadyPaidError(Exception):
    pass


class PaymentService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.ticket_repo = TicketRepository(db)
        self.payment_repo = PaymentRepository(db)
        self.event_repo = EventRepository(db)

    def pay_ticket(self, ticket_id: int, payment_method: str) -> Ticket:
        """Marca um ticket como pago, calculando o valor automaticamente.

        Raises:
            TicketNotFoundError: Se o ticket não existir.
            TicketAlreadyPaidError: Se o ticket não estiver em "EM_ABERTO"
                (já foi pago ou já foi finalizado).
        """
        ticket = self.ticket_repo.get_by_id(ticket_id)
        if ticket is None:
            raise TicketNotFoundError(f"Ticket #{ticket_id} não encontrado.")
        if ticket.status != TicketStatus.EM_ABERTO:
            raise TicketAlreadyPaidError(
                f"Ticket #{ticket_id} não está em aberto (status atual: {ticket.status.value})."
            )

        amount = self._calculate_amount(ticket.entry_at)
        now = datetime.now(timezone.utc)

        self.payment_repo.create(ticket_id=ticket.id, amount=amount, method=payment_method, paid_at=now)

        ticket.status = TicketStatus.PAGO
        ticket.amount = amount
        ticket.payment_method = payment_method
        ticket.paid_at = now

        self.event_repo.log(
            EventType.PAYMENT, ticket_id=ticket.id, payload={"amount": amount, "method": payment_method}
        )
        self.db.commit()

        reloaded = self.ticket_repo.get_by_id(ticket.id)
        assert reloaded is not None
        logger.info(f"Ticket #{ticket.id} pago — valor=R${amount:.2f}, método={payment_method}")
        return reloaded

    @staticmethod
    def _calculate_amount(entry_at: datetime) -> float:
        now = datetime.now(timezone.utc)
        entry = entry_at if entry_at.tzinfo else entry_at.replace(tzinfo=timezone.utc)
        duration_hours = (now - entry).total_seconds() / 3600
        billed_hours = max(1, math.ceil(duration_hours))  # cobrança mínima de 1 hora
        return round(billed_hours * settings.parking_rate_per_hour, 2)
