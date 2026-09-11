"""Serviço de orquestração do fluxo de SAÍDA do estacionamento.

Fluxo (conforme especificação): captura → pipeline de IA → reconhece
placa → consulta ticket em aberto → valida pagamento → abre ou nega a
cancela (simulada) → finaliza o ticket se aprovado.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.core.logging import get_logger
from backend.app.models.event import EventType
from backend.app.models.ticket import Ticket, TicketStatus
from backend.app.repositories.event_repository import EventRepository
from backend.app.repositories.ticket_repository import TicketRepository
from backend.app.repositories.vehicle_repository import VehicleRepository
from backend.app.services.vision_service import VisionService

logger = get_logger(__name__)


class NoVehicleDetectedError(Exception):
    """Nenhum veículo foi detectado na imagem de saída."""


class VehicleNotRecognizedError(Exception):
    """A placa lida não corresponde a nenhum veículo com entrada registrada,
    ou não foi possível ler a placa."""


class NoOpenTicketError(Exception):
    """O veículo é conhecido, mas não tem nenhum ticket em aberto/pago
    (ex: já saiu antes, ou nunca teve entrada registrada)."""


@dataclass
class ExitResult:
    """DTO interno do serviço (não é ORM) — resultado do processamento de saída."""

    gate_open: bool
    message: str
    ticket: Optional[Ticket]


class ExitService:
    def __init__(self, db: Session, vision_service: VisionService) -> None:
        self.db = db
        self.vision_service = vision_service
        self.vehicle_repo = VehicleRepository(db)
        self.ticket_repo = TicketRepository(db)
        self.event_repo = EventRepository(db)

    def register_exit(self, image_bytes: bytes) -> ExitResult:
        """Processa uma captura de saída e decide se a cancela abre.

        Raises:
            NoVehicleDetectedError: Nenhum veículo detectado na imagem.
            VehicleNotRecognizedError: Placa não lida ou desconhecida.
            NoOpenTicketError: Veículo conhecido, mas sem ticket em aberto.
        """
        frame_result = self.vision_service.process_image_bytes(image_bytes)

        if not frame_result.vehicles:
            self.event_repo.log(EventType.DENIED, payload={"reason": "no_vehicle_detected"})
            self.db.commit()
            raise NoVehicleDetectedError("Nenhum veículo detectado na imagem enviada.")

        primary = max(frame_result.vehicles, key=lambda v: v.vehicle_detection.area)
        plate = primary.plate_text

        if not plate:
            self.event_repo.log(EventType.DENIED, payload={"reason": "plate_not_read"})
            self.db.commit()
            raise VehicleNotRecognizedError("Não foi possível ler a placa na saída.")

        vehicle = self.vehicle_repo.get_by_plate(plate)
        if vehicle is None:
            self.event_repo.log(EventType.DENIED, payload={"reason": "unknown_plate", "plate": plate})
            self.db.commit()
            raise VehicleNotRecognizedError(
                f"Placa {plate} não corresponde a nenhum veículo com entrada registrada."
            )

        ticket = self.ticket_repo.get_open_ticket_by_vehicle(vehicle.id)
        if ticket is None:
            self.event_repo.log(EventType.DENIED, payload={"reason": "no_open_ticket", "plate": plate})
            self.db.commit()
            raise NoOpenTicketError(f"Nenhum ticket em aberto encontrado para a placa {plate}.")

        if ticket.status != TicketStatus.PAGO:
            self.event_repo.log(EventType.DENIED, ticket_id=ticket.id, payload={"reason": "payment_pending"})
            self.db.commit()
            reloaded = self.ticket_repo.get_by_id(ticket.id)
            logger.info(f"Saída negada — Ticket #{ticket.id} com pagamento pendente.")
            return ExitResult(gate_open=False, message="Pagamento pendente.", ticket=reloaded)

        ticket.exit_at = datetime.now(timezone.utc)
        ticket.status = TicketStatus.FINALIZADO
        self.event_repo.log(EventType.EXIT, ticket_id=ticket.id, payload={"plate": plate})
        self.db.commit()

        reloaded = self.ticket_repo.get_by_id(ticket.id)
        logger.info(f"Ticket #{ticket.id} finalizado — saída registrada, cancela aberta.")
        return ExitResult(gate_open=True, message="Cancela aberta.", ticket=reloaded)
