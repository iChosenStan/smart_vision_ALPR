"""DTOs Pydantic relacionados a Payment."""

from __future__ import annotations

from pydantic import BaseModel


class PaymentRequest(BaseModel):
    """Corpo do PUT /api/tickets/{id}/pay.

    O valor (`amount`) NUNCA é enviado pelo cliente — é sempre calculado
    no servidor a partir do tempo de permanência, evitando que o
    front-end possa manipular o valor cobrado (mesmo sendo uma simulação).
    """

    payment_method: str = "dinheiro"
