"""Endpoint de SAÍDA do estacionamento."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.exit import ExitResponse
from backend.app.schemas.ticket import TicketRead
from backend.app.services.exit_service import (
    ExitService,
    NoOpenTicketError,
    NoVehicleDetectedError,
    VehicleNotRecognizedError,
)

router = APIRouter(tags=["exit"])


@router.post("/exit", response_model=ExitResponse)
async def register_exit(
    request: Request,
    file: UploadFile,
    db: Session = Depends(get_db),
) -> ExitResponse:
    """Recebe uma imagem (captura simulada da câmera de saída), reconhece
    o veículo e decide se a cancela (simulada) abre.

    Retorna 200 mesmo quando a cancela permanece fechada por pagamento
    pendente — essa é uma decisão de negócio válida, não um erro.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")

    vision_service = request.app.state.vision_service
    service = ExitService(db=db, vision_service=vision_service)

    try:
        result = service.register_exit(image_bytes)
    except NoVehicleDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (VehicleNotRecognizedError, NoOpenTicketError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExitResponse(
        gate_open=result.gate_open,
        message=result.message,
        ticket=TicketRead.model_validate(result.ticket) if result.ticket else None,
    )
