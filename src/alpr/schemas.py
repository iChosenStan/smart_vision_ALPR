"""Modelos de dados do módulo de detecção/reconhecimento de placa (ALPR)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PlateDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.bbox_xyxy
        return x2 - x1

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.bbox_xyxy
        return y2 - y1


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    normalized_text: str
    matches_known_format: bool
