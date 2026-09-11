"""Serviço de orquestração do fluxo de ENTRADA do estacionamento.

Fluxo (conforme especificação): captura → pipeline de IA (veículo → placa
→ OCR → classificação) → salva imagens → cria/recupera veículo → cria
ticket "EM_ABERTO" → registra evento → retorna o ticket (exibido como
"ticket virtual" no frontend).

DECISÃO DE NEGÓCIO TOMADA AQUI (não estava especificada, documentando
para validação): se um veículo é detectado mas a placa não é lida pelo
OCR (baixa confiança, obstrução, ângulo ruim etc.), a entrada NÃO é
bloqueada — um identificador temporário (`DESCONHECIDA-XXXXXXXX`) é usado
no lugar, com `ocr_confidence=0.0`, sinalizando a necessidade de revisão
manual. Se preferir bloquear a entrada nesse caso (cancela não abre),
me avisa que ajusto.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.models.event import EventType
from backend.app.models.ticket import Ticket
from backend.app.repositories.event_repository import EventRepository
from backend.app.repositories.ticket_repository import TicketRepository
from backend.app.repositories.vehicle_repository import VehicleRepository
from backend.app.services.vision_service import VisionService

logger = get_logger(__name__)


class NoVehicleDetectedError(Exception):
    """Levantado quando o pipeline de IA não encontra nenhum veículo na imagem."""


class EntryService:
    def __init__(self, db: Session, vision_service: VisionService) -> None:
        self.db = db
        self.vision_service = vision_service
        self.vehicle_repo = VehicleRepository(db)
        self.ticket_repo = TicketRepository(db)
        self.event_repo = EventRepository(db)

    def register_entry(self, image_bytes: bytes) -> Ticket:
        """Processa uma captura de entrada e cria o ticket correspondente.

        Raises:
            NoVehicleDetectedError: Se nenhum veículo for detectado na imagem.
        """
        frame_result = self.vision_service.process_image_bytes(image_bytes)

        if not frame_result.vehicles:
            self.event_repo.log(EventType.DENIED, payload={"reason": "no_vehicle_detected"})
            self.db.commit()
            raise NoVehicleDetectedError("Nenhum veículo detectado na imagem enviada.")

        # Assume 1 veículo principal por captura (mesma premissa documentada
        # na avaliação end-to-end da Etapa 8): usa o de maior bbox.
        primary = max(frame_result.vehicles, key=lambda v: v.vehicle_detection.area)

        plate = primary.plate_text
        ocr_confidence = primary.ocr_result.confidence if primary.ocr_result else 0.0
        if not plate:
            plate = f"DESCONHECIDA-{uuid.uuid4().hex[:8].upper()}"
            logger.warning(f"OCR não leu a placa — usando identificador temporário: {plate}")

        vehicle = self.vehicle_repo.get_or_create(
            plate=plate,
            make=primary.classification.make.label,
            model=primary.classification.model.label,
            color=primary.classification.color.label,
            type_=primary.classification.type.label,
        )

        entry_image_path = self._save_crop(primary.vehicle_crop, "vehicles")
        plate_image_path = self._save_crop(primary.plate_crop, "plates")

        ticket = self.ticket_repo.create(
            vehicle_id=vehicle.id,
            entry_image_path=entry_image_path,
            plate_image_path=plate_image_path,
            ocr_confidence=ocr_confidence,
        )

        self.event_repo.log(EventType.ENTRY, ticket_id=ticket.id, payload={"plate": plate})
        self.db.commit()

        # Recarrega já com `vehicle` carregado via joinedload, evitando
        # erro de sessão fechada ao serializar a resposta HTTP.
        reloaded = self.ticket_repo.get_by_id(ticket.id)
        assert reloaded is not None
        logger.info(f"Ticket #{reloaded.id} criado — placa={plate}, confiança OCR={ocr_confidence:.2f}")
        return reloaded

    def _save_crop(self, crop: Optional[np.ndarray], subfolder: str) -> Optional[str]:
        """Salva um recorte de imagem em disco e retorna o caminho relativo
        (relativo a `settings.captures_dir`) a ser guardado no banco.
        """
        if crop is None or crop.size == 0:
            return None

        filename = f"{uuid.uuid4().hex}.jpg"
        dest_dir = settings.captures_dir / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        cv2.imwrite(str(dest_path), crop)
        return str(Path(subfolder) / filename)
