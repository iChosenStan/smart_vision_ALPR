"""Endpoint de ENTRADA do estacionamento."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.ticket import TicketRead
from backend.app.services.entry_service import EntryService, NoVehicleDetectedError

router = APIRouter(tags=["entry"])


@router.post("/entry", response_model=TicketRead, status_code=status.HTTP_201_CREATED)
async def register_entry(
    request: Request,
    file: UploadFile,
    db: Session = Depends(get_db),
) -> TicketRead:
    """Recebe uma imagem (captura simulada da câmera de entrada), roda o
    pipeline de IA completo e cria o ticket correspondente.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")

    vision_service = request.app.state.vision_service
    service = EntryService(db=db, vision_service=vision_service)

    try:
        ticket = service.register_entry(image_bytes)
    except NoVehicleDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:  # imagem enviada não pôde ser decodificada
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TicketRead.model_validate(ticket)
