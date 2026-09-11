"""Fixtures compartilhados entre os testes do backend.

Centraliza a construção de um `VisionService` fake (usando os dataclasses
REAIS e leves de `src.pipeline.schemas`) e do `TestClient` da aplicação,
evitando duplicação entre `test_entry.py`, `test_payment.py`, `test_exit.py`, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core import database as database_module
from backend.app.services.vision_service import VisionService
from src.alpr.schemas import OCRResult, PlateDetection
from src.classification.schemas import AttributePrediction, VehicleClassification
from src.detection.schemas import VehicleDetection
from src.pipeline.schemas import FrameResult, VehicleResult


def make_vehicle_result(plate: Optional[str], confidence: float = 0.9) -> VehicleResult:
    ocr_result = None
    if plate:
        ocr_result = OCRResult(
            text=plate, confidence=confidence, normalized_text=plate, matches_known_format=True
        )

    return VehicleResult(
        vehicle_detection=VehicleDetection(
            bbox_xyxy=(10, 10, 210, 210), confidence=0.95, class_id=2, class_name="car"
        ),
        plate_detection=PlateDetection(bbox_xyxy=(5, 5, 60, 25), confidence=0.8, class_id=0, class_name="plate"),
        ocr_result=ocr_result,
        classification=VehicleClassification(
            type=AttributePrediction(label="car", confidence=0.9),
            make=AttributePrediction(label="fiat", confidence=0.8),
            model=AttributePrediction(label="uno", confidence=0.7),
            color=AttributePrediction(label="white", confidence=0.6),
        ),
        plate_detection_ms=5.0,
        ocr_ms=8.0,
        classification_ms=6.0,
        vehicle_crop=np.zeros((200, 200, 3), dtype=np.uint8),
        plate_crop=np.zeros((20, 55, 3), dtype=np.uint8) if plate else None,
    )


class _FakeVisionService:
    def __init__(self, frame_result: FrameResult) -> None:
        self._frame_result = frame_result

    def process_image_bytes(self, image_bytes: bytes) -> FrameResult:
        return self._frame_result


class _ReplayVisionService:
    """Retorna um FrameResult diferente a cada chamada, na ordem dada —
    necessário para testar fluxos com múltiplos veículos/placas distintas
    (ex: entrada de 3 veículos diferentes, depois listagem/dashboard)."""

    def __init__(self, frame_results: List[FrameResult]) -> None:
        self._queue = list(frame_results)

    def process_image_bytes(self, image_bytes: bytes) -> FrameResult:
        return self._queue.pop(0)


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    """Fixture-fábrica: cada teste escolhe qual FrameResult o fake retorna."""

    def _make(frame_result: FrameResult) -> TestClient:
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/test.db")
        monkeypatch.setattr(settings, "captures_dir", tmp_path / "captures")
        database_module.reset_engine_for_testing()

        fake_service = _FakeVisionService(frame_result)
        monkeypatch.setattr(VisionService, "load", classmethod(lambda cls, *a, **kw: fake_service))

        from backend.app.main import app

        return TestClient(app)

    return _make


@pytest.fixture
def make_client_sequence(monkeypatch, tmp_path):
    """Fixture-fábrica: aceita uma LISTA de FrameResults, um por chamada
    consecutiva ao pipeline (útil para simular múltiplos veículos)."""

    def _make(frame_results: List[FrameResult]) -> TestClient:
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/test.db")
        monkeypatch.setattr(settings, "captures_dir", tmp_path / "captures")
        database_module.reset_engine_for_testing()

        fake_service = _ReplayVisionService(frame_results)
        monkeypatch.setattr(VisionService, "load", classmethod(lambda cls, *a, **kw: fake_service))

        from backend.app.main import app

        return TestClient(app)

    return _make
