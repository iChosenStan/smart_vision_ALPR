"""Testes da Etapa 3 — detecção de veículos.

Usa mocks para a saída do `ultralytics.YOLO`, evitando dependência de
download real de pesos (~6MB, indisponível em ambientes de CI/sandbox
com rede restrita) e mantendo os testes rápidos e determinísticos.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.detection.schemas import VehicleDetection
from src.detection.vehicle_detector import VehicleDetector
from src.utils.exceptions import ModelLoadError, VehicleDetectionError


class _FakeBoxes:
    """Mock mínimo compatível com a interface `ultralytics.engine.results.Boxes`."""

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
    """Mock do `ultralytics.YOLO` — permite controlar a saída de `predict()`."""

    def __init__(self, result: _FakeResult) -> None:
        self._result = result
        self.last_predict_kwargs = None

    def predict(self, image, **kwargs):
        self.last_predict_kwargs = kwargs
        return [self._result]


def _make_detector(fake_result: _FakeResult, **overrides) -> VehicleDetector:
    fake_model = _FakeYOLOModel(fake_result)
    kwargs = dict(device="cpu", _model=fake_model)
    kwargs.update(overrides)
    return VehicleDetector(**kwargs)


def test_detect_filters_out_non_vehicle_classes() -> None:
    # classes COCO: 0=person, 2=car, 16=dog — apenas "car" deve sobreviver
    boxes = _FakeBoxes(
        cls=[0, 2, 16],
        conf=[0.9, 0.87, 0.6],
        xyxy=[[10, 10, 50, 50], [100, 100, 300, 250], [5, 5, 20, 20]],
    )
    detector = _make_detector(_FakeResult(boxes))
    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(fake_image)

    assert len(detections) == 1
    assert isinstance(detections[0], VehicleDetection)
    assert detections[0].class_name == "car"
    assert detections[0].class_id == 2
    assert detections[0].confidence == pytest.approx(0.87)
    assert detections[0].bbox_xyxy == (100.0, 100.0, 300.0, 250.0)


def test_detect_keeps_all_default_vehicle_classes() -> None:
    # car=2, motorcycle=3, bus=5, truck=7
    boxes = _FakeBoxes(
        cls=[2, 3, 5, 7],
        conf=[0.9, 0.8, 0.7, 0.6],
        xyxy=[[0, 0, 10, 10]] * 4,
    )
    detector = _make_detector(_FakeResult(boxes))
    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert {d.class_name for d in detections} == {"car", "motorcycle", "bus", "truck"}


def test_detect_returns_empty_list_when_no_boxes() -> None:
    boxes = _FakeBoxes(cls=[], conf=[], xyxy=[])
    detector = _make_detector(_FakeResult(boxes))
    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert detections == []


def test_detect_respects_custom_target_classes() -> None:
    boxes = _FakeBoxes(
        cls=[2, 5],
        conf=[0.9, 0.8],
        xyxy=[[0, 0, 10, 10], [0, 0, 20, 20]],
    )
    detector = _make_detector(_FakeResult(boxes), target_classes={5: "bus"})
    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].class_name == "bus"


def test_detect_passes_inference_params_to_model() -> None:
    boxes = _FakeBoxes(cls=[], conf=[], xyxy=[])
    fake_model = _FakeYOLOModel(_FakeResult(boxes))
    detector = VehicleDetector(
        device="cpu",
        confidence_threshold=0.5,
        iou_threshold=0.4,
        image_size=320,
        _model=fake_model,
    )
    detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert fake_model.last_predict_kwargs["conf"] == 0.5
    assert fake_model.last_predict_kwargs["iou"] == 0.4
    assert fake_model.last_predict_kwargs["imgsz"] == 320
    assert fake_model.last_predict_kwargs["device"] == "cpu"


def test_detect_raises_vehicle_detection_error_on_model_failure() -> None:
    class _BrokenModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("falha simulada de inferência")

    detector = VehicleDetector(device="cpu", _model=_BrokenModel())
    with pytest.raises(VehicleDetectionError):
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    boxes = _FakeBoxes(cls=[], conf=[], xyxy=[])
    detector = _make_detector(_FakeResult(boxes), device="cuda")
    assert detector.device == "cpu"


def test_vehicle_detection_derived_properties() -> None:
    det = VehicleDetection(bbox_xyxy=(10.0, 20.0, 110.0, 70.0), confidence=0.9, class_id=2, class_name="car")
    assert det.width == 100.0
    assert det.height == 50.0
    assert det.area == 5000.0


def test_from_config_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(ModelLoadError):
        VehicleDetector.from_config(tmp_path / "nao_existe.yaml")


def test_from_config_loads_expected_parameters(monkeypatch) -> None:
    """Valida que from_config lê corretamente configs/detection.yaml,
    sem baixar pesos reais (substitui _load_model por um mock)."""
    boxes = _FakeBoxes(cls=[], conf=[], xyxy=[])
    fake_model = _FakeYOLOModel(_FakeResult(boxes))
    monkeypatch.setattr(VehicleDetector, "_load_model", staticmethod(lambda weights, device: fake_model))

    detector = VehicleDetector.from_config("configs/detection.yaml")

    assert detector.weights == "yolov8n.pt"
    assert detector.confidence_threshold == 0.35
    assert detector.iou_threshold == 0.45
    assert detector.image_size == 640
    assert detector.target_classes == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
