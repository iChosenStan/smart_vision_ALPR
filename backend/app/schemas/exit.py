"""DTOs Pydantic relacionados ao fluxo de saída."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from backend.app.schemas.ticket import TicketRead


class ExitResponse(BaseModel):
    """Resposta do POST /api/exit — inclui o estado da cancela simulada."""

    gate_open: bool
    message: str
    ticket: Optional[TicketRead] = None
