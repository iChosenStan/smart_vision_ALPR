"""Testes da Etapa 4 — detecção de placa (modelo fine-tunado).

Usa mocks para a saída do `ultralytics.YOLO`, seguindo o mesmo padrão de
`tests/test_vehicle_detector.py`. Também valida que `PlateDetector` exige
que os pesos existam em disco (diferente de `VehicleDetector`, que baixa
automaticamente).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.alpr.plate_detector import PlateDetector
from src.alpr.schemas import PlateDetection
from src.utils.exceptions import ModelLoadError, PlateDetectionError


class _FakeBoxes:
    def __init__(self, cls: List[int], conf: List[float], xyxy: List[List[float]]) -> None:
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy

    def __len__(self) -> int:
        return len(self.cls)


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _FakeYOLOModel:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result

    def predict(self, image, **kwargs):
        return [self._result]


def _make_detector(fake_result: _FakeResult, **overrides) -> PlateDetector:
    fake_model = _FakeYOLOModel(fake_result)
    kwargs = dict(device="cpu", _model=fake_model)
    kwargs.update(overrides)
    return PlateDetector(**kwargs)


def test_detect_returns_plate_detection_instances() -> None:
    boxes = _FakeBoxes(cls=[0], conf=[0.92], xyxy=[[120, 300, 260, 340]])
    detector = _make_detector(_FakeResult(boxes))

    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert isinstance(detections[0], PlateDetection)
    assert detections[0].class_name == "plate"
    assert detections[0].confidence == pytest.approx(0.92)
    assert detections[0].bbox_xyxy == (120.0, 300.0, 260.0, 340.0)


def test_detect_returns_empty_list_when_no_plate_found() -> None:
    boxes = _FakeBoxes(cls=[], conf=[], xyxy=[])
    detector = _make_detector(_FakeResult(boxes))
    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert detections == []


def test_detect_raises_plate_detection_error_on_model_failure() -> None:
    class _BrokenModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("falha simulada")

    detector = PlateDetector(device="cpu", _model=_BrokenModel())
    with pytest.raises(PlateDetectionError):
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_load_model_raises_when_weights_file_does_not_exist(tmp_path) -> None:
    missing_weights = tmp_path / "nao_existe.pt"
    with pytest.raises(ModelLoadError):
        PlateDetector(weights=str(missing_weights), device="cpu")


def test_crop_best_plate_returns_none_when_no_detections() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert PlateDetector.crop_best_plate(image, []) is None


def test_crop_best_plate_returns_highest_confidence_crop() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[10:20, 10:30] = 255  # marca região de interesse (baixa confiança)
    image[50:60, 50:90] = 128  # marca região de interesse (alta confiança)

    low_conf = PlateDetection(bbox_xyxy=(10, 10, 30, 20), confidence=0.4, class_id=0, class_name="plate")
    high_conf = PlateDetection(bbox_xyxy=(50, 50, 90, 60), confidence=0.9, class_id=0, class_name="plate")

    crop = PlateDetector.crop_best_plate(image, [low_conf, high_conf])

    assert crop is not None
    assert crop.shape == (10, 40, 3)  # (y2-y1, x2-x1, canais) da detecção de maior confiança
    assert (crop == 128).all()


def test_from_config_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(ModelLoadError):
        PlateDetector.from_config(tmp_path / "nao_existe.yaml")


def test_from_config_loads_expected_parameters(monkeypatch, tmp_path) -> None:
    """Valida leitura de configs/plate_detection.yaml sem carregar pesos reais."""
    boxes = _FakeBoxes(cls=[], conf=[], xyxy=[])
    fake_model = _FakeYOLOModel(_FakeResult(boxes))
    monkeypatch.setattr(PlateDetector, "_load_model", staticmethod(lambda weights, device: fake_model))

    detector = PlateDetector.from_config("configs/plate_detection.yaml")

    assert detector.weights == "models/checkpoints/plate_detector_best.pt"
    assert detector.confidence_threshold == 0.40
    assert detector.iou_threshold == 0.45
    assert detector.image_size == 640
    assert detector.target_classes == {0: "plate"}
