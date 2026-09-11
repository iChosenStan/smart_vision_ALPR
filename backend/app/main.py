"""Ponto de entrada da API do SmartVision ALPR — Estacionamento Inteligente.

Os 4 modelos de IA (veículo, placa, OCR, classificação) são carregados
UMA ÚNICA VEZ no startup (`lifespan`) e mantidos em memória durante toda a
vida da aplicação — nunca recarregados por requisição.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routers import dashboard, entry, exit, health, history, tickets
from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.core.logging import get_logger
from backend.app.services.vision_service import VisionService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando SmartVision ALPR API...")
    settings.captures_dir.mkdir(parents=True, exist_ok=True)
    init_db()

    app.state.vision_service = VisionService.load()

    logger.info("SmartVision ALPR API pronta.")
    yield
    logger.info("Encerrando SmartVision ALPR API.")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(entry.router, prefix="/api")
app.include_router(exit.router, prefix="/api")
app.include_router(tickets.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(history.router, prefix="/api")

# Serve as imagens capturadas (foto do veículo/placa de cada ticket) para
# o frontend exibir — ex: http://localhost:8000/captures/vehicles/xxx.jpg
settings.captures_dir.mkdir(parents=True, exist_ok=True)
app.mount("/captures", StaticFiles(directory=str(settings.captures_dir)), name="captures")
