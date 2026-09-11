"""Modelos de dados do pipeline de inferência integrado (Etapa 7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.alpr.schemas import OCRResult, PlateDetection
from src.classification.schemas import VehicleClassification
from src.detection.schemas import VehicleDetection

# Meta de latência definida em configs/base.yaml (pipeline_targets.inference_latency_ms_max)
_LATENCY_TARGET_MS = 100.0


@dataclass
class VehicleResult:
    """Resultado completo do pipeline para um único veículo detectado no frame."""

    vehicle_detection: VehicleDetection
    plate_detection: Optional[PlateDetection]
    ocr_result: Optional[OCRResult]
    classification: VehicleClassification
    plate_detection_ms: float
    ocr_ms: float
    classification_ms: float
    # NOVO (backend Módulo 2): recortes de imagem, necessários para o
    # sistema de estacionamento salvar "foto do veículo" e "foto da placa"
    # por ticket. Opcionais (default None) para não quebrar código
    # existente que constrói VehicleResult sem esses campos.
    vehicle_crop: Optional[np.ndarray] = None
    plate_crop: Optional[np.ndarray] = None

    @property
    def plate_text(self) -> Optional[str]:
        """Atalho para o texto normalizado da placa, se reconhecida."""
        return self.ocr_result.normalized_text if self.ocr_result else None


@dataclass
class FrameResult:
    """Resultado completo do pipeline para um frame (pode conter 0+ veículos)."""

    vehicles: List[VehicleResult] = field(default_factory=list)
    vehicle_detection_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0

    @property
    def meets_latency_target(self) -> bool:
        """Verifica se o frame atendeu a meta de latência do projeto (100ms)."""
        return self.total_latency_ms <= _LATENCY_TARGET_MS
