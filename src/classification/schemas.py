"""Modelos de dados do módulo de classificação de veículo."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttributePrediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class VehicleClassification:
    type: AttributePrediction
    make: AttributePrediction
    model: AttributePrediction
    color: AttributePrediction
