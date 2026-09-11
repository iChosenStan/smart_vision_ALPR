"""Agrega todos os modelos ORM — importado por `database.init_db()` para
garantir que o SQLAlchemy registre todas as tabelas antes do `create_all()`.
"""

from backend.app.models.event import Event, EventType
from backend.app.models.log import Log, LogSource
from backend.app.models.payment import Payment
from backend.app.models.ticket import Ticket, TicketStatus
from backend.app.models.vehicle import Vehicle

__all__ = [
    "Vehicle",
    "Ticket",
    "TicketStatus",
    "Payment",
    "Event",
    "EventType",
    "Log",
    "LogSource",
]
