"""Testes da Etapa 8 — rotinas de avaliação (evaluator.py).

Usa componentes *fake* (mesmo padrão de duck typing da Etapa 7) e imagens
sintéticas, validando 100% da lógica de avaliação sem depender de nenhum
peso real.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import pytest

from src.alpr.schemas import OCRResult
from src.classification.schemas import AttributePrediction, VehicleClassification
from src.detection.schemas import VehicleDetection
from src.evaluation.evaluator import (
    evaluate_classifier_isolated,
    evaluate_ocr_isolated,
    evaluate_pipeline_end_to_end,
)
from src.evaluation.schemas import ClassificationMetrics, EndToEndMetrics, OCRMetrics
from src.pipeline.schemas import FrameResult, VehicleResult


def _write_dummy_image(path: Path, size=(300, 400)) -> None:
    h, w = size
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


@pytest.fixture
def images_dir(tmp_path) -> Path:
    d = tmp_path / "images"
    d.mkdir()
    _write_dummy_image(d / "img_a.jpg")
    _write_dummy_image(d / "img_b.jpg")
    return d


def _make_test_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "filename": "img_a.jpg", "plate": "ABC1234", "type": "car", "make": "fiat",
                "model": "uno", "color": "white",
                "plate_bbox_x1": 50, "plate_bbox_y1": 50, "plate_bbox_x2": 150, "plate_bbox_y2": 100,
            },
            {
                "filename": "img_b.jpg", "plate": "DEF5678", "type": "truck", "make": "ford",
                "model": "f250", "color": "red",
                "plate_bbox_x1": 30, "plate_bbox_y1": 40, "plate_bbox_x2": 120, "plate_bbox_y2": 90,
            },
            {
                "filename": "img_missing.jpg", "plate": "GHI9999", "type": "car", "make": "vw",
                "model": "gol", "color": "black",
                "plate_bbox_x1": 10, "plate_bbox_y1": 10, "plate_bbox_x2": 40, "plate_bbox_y2": 40,
            },
        ]
    )


class _FakeOCRReader:
    """Retorna a placa correta para img_a, uma placa errada para img_b."""

    def __init__(self) -> None:
        self.call_count = 0

    def read(self, crop) -> Optional[OCRResult]:
        self.call_count += 1
        if self.call_count == 1:
            return OCRResult(text="ABC1234", confidence=0.9, normalized_text="ABC1234", matches_known_format=True)
        return OCRResult(text="XXX0000", confidence=0.5, normalized_text="XXX0000", matches_known_format=True)


class _FakeVehicleClassifier:
    """Acerta type/make sempre, erra model/color sempre (para validar
    que as métricas por atributo são calculadas de forma independente).
    """

    def classify(self, image) -> VehicleClassification:
        return VehicleClassification(
            type=AttributePrediction(label="car", confidence=0.9),
            make=AttributePrediction(label="fiat", confidence=0.8),
            model=AttributePrediction(label="ERRADO", confidence=0.5),
            color=AttributePrediction(label="ERRADO", confidence=0.5),
        )


class _FakePipeline:
    """Simula o SmartVisionPipeline: detecta veículo só na 1ª chamada."""

    def __init__(self) -> None:
        self.call_count = 0

    def process_frame(self, frame) -> FrameResult:
        self.call_count += 1
        if self.call_count == 1:
            vehicle_result = VehicleResult(
                vehicle_detection=VehicleDetection(bbox_xyxy=(0, 0, 100, 100), confidence=0.9, class_id=2, class_name="car"),
                plate_detection=None,
                ocr_result=OCRResult(text="ABC1234", confidence=0.9, normalized_text="ABC1234", matches_known_format=True),
                classification=VehicleClassification(
                    type=AttributePrediction(label="car", confidence=0.9),
                    make=AttributePrediction(label="fiat", confidence=0.8),
                    model=AttributePrediction(label="uno", confidence=0.7),
                    color=AttributePrediction(label="white", confidence=0.6),
                ),
                plate_detection_ms=5.0, ocr_ms=10.0, classification_ms=8.0,
            )
            return FrameResult(vehicles=[vehicle_result], vehicle_detection_ms=20.0, total_latency_ms=50.0, fps=20.0)
        # 2ª chamada: nenhum veículo detectado
        return FrameResult(vehicles=[], vehicle_detection_ms=15.0, total_latency_ms=15.0, fps=66.0)


def test_evaluate_ocr_isolated_computes_expected_metrics(images_dir) -> None:
    manifest = _make_test_manifest()
    ocr_reader = _FakeOCRReader()

    metrics = evaluate_ocr_isolated(manifest, images_dir, ocr_reader, margin_ratio=0.0)

    assert isinstance(metrics, OCRMetrics)
    # img_a e img_b existem (2 amostras processadas), img_missing.jpg é pulada
    assert metrics.num_samples == 2
    # 1 acerto (img_a) em 2 -> exact-match = 50%
    assert metrics.exact_match_accuracy == pytest.approx(0.5)


def test_evaluate_ocr_isolated_meets_target_property() -> None:
    metrics = OCRMetrics(num_samples=10, exact_match_accuracy=0.96, character_accuracy=0.98, known_format_rate=1.0)
    assert metrics.meets_target(0.95) is True
    assert metrics.meets_target(0.97) is False


def test_evaluate_classifier_isolated_computes_per_attribute_metrics(images_dir) -> None:
    manifest = _make_test_manifest()
    classifier = _FakeVehicleClassifier()

    metrics = evaluate_classifier_isolated(manifest, images_dir, classifier)

    assert isinstance(metrics, ClassificationMetrics)
    # "type" e "make" sempre acertam (fake sempre retorna car/fiat, mas
    # img_b tem ground truth truck/ford) -> vamos conferir valores reais:
    # img_a: type=car(GT car) OK, make=fiat(GT fiat) OK
    # img_b: type=car(GT truck) ERRO, make=fiat(GT ford) ERRO
    assert metrics.per_attribute["type"].accuracy == pytest.approx(0.5)
    assert metrics.per_attribute["model"].accuracy == 0.0  # sempre "ERRADO"
    assert metrics.per_attribute["color"].accuracy == 0.0


def test_evaluate_pipeline_end_to_end_handles_missing_detection(images_dir) -> None:
    manifest = _make_test_manifest()
    pipeline = _FakePipeline()

    metrics = evaluate_pipeline_end_to_end(manifest, images_dir, pipeline)

    assert isinstance(metrics, EndToEndMetrics)
    assert metrics.num_samples == 2  # img_missing.jpg pulada
    # 1 detecção bem-sucedida em 2 chamadas
    assert metrics.vehicle_detection_rate == pytest.approx(0.5)
    # a única detecção teve OCR correto (ABC1234) -> 1/2 exact-match
    assert metrics.plate_exact_match_accuracy == pytest.approx(0.5)


def test_evaluate_pipeline_end_to_end_computes_latency_stats(images_dir) -> None:
    manifest = _make_test_manifest()
    pipeline = _FakePipeline()

    metrics = evaluate_pipeline_end_to_end(manifest, images_dir, pipeline)

    # latências das 2 chamadas: 50.0 e 15.0 -> média = 32.5
    assert metrics.avg_latency_ms == pytest.approx(32.5)
    assert metrics.meets_latency_target(100.0) is True
    assert metrics.meets_latency_target(10.0) is False
