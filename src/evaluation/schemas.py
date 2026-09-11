"""Modelos de dados dos relatórios de avaliação (Etapa 8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class OCRMetrics:
    """Resultado da avaliação isolada do OCR (recorte de placa via ground-truth)."""

    num_samples: int
    exact_match_accuracy: float  # métrica oficial (meta: ≥95%)
    character_accuracy: float  # diagnóstico complementar
    known_format_rate: float  # % de predições que batem um padrão de placa conhecido

    def meets_target(self, target: float = 0.95) -> bool:
        return self.exact_match_accuracy >= target


@dataclass
class AttributeMetrics:
    """Métricas de um único atributo de classificação (ex: apenas `color`)."""

    num_samples: int
    accuracy: float  # métrica oficial (meta: ≥80%)
    balanced_accuracy: float  # diagnóstico complementar
    num_classes: int

    def meets_target(self, target: float = 0.80) -> bool:
        return self.accuracy >= target


@dataclass
class ClassificationMetrics:
    """Resultado da avaliação isolada do classificador (4 atributos)."""

    per_attribute: Dict[str, AttributeMetrics] = field(default_factory=dict)

    @property
    def mean_accuracy(self) -> float:
        if not self.per_attribute:
            return 0.0
        return sum(m.accuracy for m in self.per_attribute.values()) / len(self.per_attribute)

    def meets_target(self, target: float = 0.80) -> bool:
        return self.mean_accuracy >= target


@dataclass
class EndToEndMetrics:
    """Resultado da avaliação ponta a ponta (pipeline completo, Etapa 7)."""

    num_samples: int
    vehicle_detection_rate: float  # fração de imagens com pelo menos 1 veículo detectado
    plate_exact_match_accuracy: float  # considera detecção ausente como erro
    classification_mean_accuracy: float
    avg_latency_ms: float
    avg_fps: float

    def meets_latency_target(self, target_ms: float = 100.0) -> bool:
        return self.avg_latency_ms <= target_ms
