"""Testes da Etapa 7 — pipeline integrado (SmartVisionPipeline).

Usa componentes *fake* (duck typing — mesma interface de
VehicleDetector/PlateDetector/OCRReader/VehicleClassifier), validando
100% da lógica de orquestração (recorte, roteamento, agregação de
resultados, medição de latência) sem depender de nenhum peso real.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from src.alpr.schemas import OCRResult, PlateDetection
from src.classification.schemas import AttributePrediction, VehicleClassification
from src.detection.schemas import VehicleDetection
from src.pipeline.smart_vision_pipeline import SmartVisionPipeline

_DUMMY_CLASSIFICATION = VehicleClassification(
    type=AttributePrediction(label="car", confidence=0.9),
    make=AttributePrediction(label="fiat", confidence=0.8),
    model=AttributePrediction(label="uno", confidence=0.7),
    color=AttributePrediction(label="white", confidence=0.6),
)


class _FakeVehicleDetector:
    def __init__(self, detections: List[VehicleDetection]) -> None:
        self._detections = detections
        self.received_frames: List[np.ndarray] = []

    def detect(self, frame) -> List[VehicleDetection]:
        self.received_frames.append(frame)
        return self._detections


class _FakePlateDetector:
    def __init__(self, plates_by_call: List[List[PlateDetection]]) -> None:
        self._plates_by_call = plates_by_call
        self._call_idx = 0
        self.received_crops: List[np.ndarray] = []

    def detect(self, crop) -> List[PlateDetection]:
        self.received_crops.append(crop)
        result = self._plates_by_call[self._call_idx]
        self._call_idx += 1
        return result


class _FakeOCRReader:
    def __init__(self, result: Optional[OCRResult]) -> None:
        self._result = result
        self.received_crops: List[np.ndarray] = []

    def read(self, crop) -> Optional[OCRResult]:
        self.received_crops.append(crop)
        return self._result


class _FakeVehicleClassifier:
    def __init__(self, result: VehicleClassification = _DUMMY_CLASSIFICATION) -> None:
        self._result = result
        self.received_crops: List[np.ndarray] = []

    def classify(self, crop) -> VehicleClassification:
        self.received_crops.append(crop)
        return self._result


def _make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_vehicle_detection(bbox=(100.0, 100.0, 300.0, 300.0)) -> VehicleDetection:
    return VehicleDetection(bbox_xyxy=bbox, confidence=0.9, class_id=2, class_name="car")


def test_process_frame_with_single_vehicle_and_plate_found() -> None:
    vehicle_det = _make_vehicle_detection()
    plate_det = PlateDetection(bbox_xyxy=(10, 10, 60, 30), confidence=0.8, class_id=0, class_name="plate")
    ocr_result = OCRResult(text="ABC1234", confidence=0.95, normalized_text="ABC1234", matches_known_format=True)

    pipeline = SmartVisionPipeline(
        vehicle_detector=_FakeVehicleDetector([vehicle_det]),
        plate_detector=_FakePlateDetector([[plate_det]]),
        ocr_reader=_FakeOCRReader(ocr_result),
        vehicle_classifier=_FakeVehicleClassifier(),
    )

    result = pipeline.process_frame(_make_frame())

    assert len(result.vehicles) == 1
    v = result.vehicles[0]
    assert v.plate_text == "ABC1234"
    assert v.classification.make.label == "fiat"
    assert result.total_latency_ms >= 0
    assert result.fps > 0


def test_process_frame_handles_vehicle_without_plate_found() -> None:
    vehicle_det = _make_vehicle_detection()

    pipeline = SmartVisionPipeline(
        vehicle_detector=_FakeVehicleDetector([vehicle_det]),
        plate_detector=_FakePlateDetector([[]]),  # nenhuma placa detectada
        ocr_reader=_FakeOCRReader(None),
        vehicle_classifier=_FakeVehicleClassifier(),
    )

    result = pipeline.process_frame(_make_frame())

    assert len(result.vehicles) == 1
    v = result.vehicles[0]
    assert v.plate_detection is None
    assert v.ocr_result is None
    assert v.plate_text is None
    # classificação deve rodar mesmo sem placa encontrada
    assert v.classification is not None


def test_process_frame_with_multiple_vehicles() -> None:
    detections = [_make_vehicle_detection((0, 0, 100, 100)), _make_vehicle_detection((200, 200, 350, 350))]

    pipeline = SmartVisionPipeline(
        vehicle_detector=_FakeVehicleDetector(detections),
        plate_detector=_FakePlateDetector([[], []]),
        ocr_reader=_FakeOCRReader(None),
        vehicle_classifier=_FakeVehicleClassifier(),
    )

    result = pipeline.process_frame(_make_frame())
    assert len(result.vehicles) == 2


def test_process_frame_filters_vehicles_below_min_area() -> None:
    small_detection = _make_vehicle_detection((0, 0, 10, 10))  # área = 100
    large_detection = _make_vehicle_detection((0, 0, 100, 100))  # área = 10000

    pipeline = SmartVisionPipeline(
        vehicle_detector=_FakeVehicleDetector([small_detection, large_detection]),
        plate_detector=_FakePlateDetector([[]]),
        ocr_reader=_FakeOCRReader(None),
        vehicle_classifier=_FakeVehicleClassifier(),
        min_vehicle_area=5000,
    )

    result = pipeline.process_frame(_make_frame())

    assert len(result.vehicles) == 1
    assert result.vehicles[0].vehicle_detection.area == 10000


def test_process_frame_with_no_vehicles_returns_empty_result() -> None:
    pipeline = SmartVisionPipeline(
        vehicle_detector=_FakeVehicleDetector([]),
        plate_detector=_FakePlateDetector([]),
        ocr_reader=_FakeOCRReader(None),
        vehicle_classifier=_FakeVehicleClassifier(),
    )

    result = pipeline.process_frame(_make_frame())

    assert result.vehicles == []
    assert result.total_latency_ms >= 0


def test_plate_detector_and_classifier_receive_vehicle_crop_not_full_frame() -> None:
    """Valida a decisão registrada na Etapa 7: PlateDetector e
    VehicleClassifier recebem o recorte do veículo, não o frame inteiro.
    """
    vehicle_det = _make_vehicle_detection((100, 100, 300, 300))  # 200x200
    plate_detector = _FakePlateDetector([[]])
    classifier = _FakeVehicleClassifier()

    pipeline = SmartVisionPipeline(
        vehicle_detector=_FakeVehicleDetector([vehicle_det]),
        plate_detector=plate_detector,
        ocr_reader=_FakeOCRReader(None),
        vehicle_classifier=classifier,
        crop_margin_ratio=0.0,
    )

    frame = _make_frame()  # 480x640
    pipeline.process_frame(frame)

    received_crop = plate_detector.received_crops[0]
    assert received_crop.shape != frame.shape  # não é o frame inteiro
    assert received_crop.shape == (200, 200, 3)  # bbox exato (sem margem)

    received_classifier_crop = classifier.received_crops[0]
    assert received_classifier_crop.shape == (200, 200, 3)


def test_crop_margin_ratio_expands_vehicle_crop() -> None:
    vehicle_det = _make_vehicle_detection((100, 100, 200, 200))  # 100x100
    plate_detector = _FakePlateDetector([[]])

    pipeline = SmartVisionPipeline(
        vehicle_detector=_FakeVehicleDetector([vehicle_det]),
        plate_detector=plate_detector,
        ocr_reader=_FakeOCRReader(None),
        vehicle_classifier=_FakeVehicleClassifier(),
        crop_margin_ratio=0.2,
    )

    pipeline.process_frame(_make_frame())

    received_crop = plate_detector.received_crops[0]
    # 100x100 + 20% de margem de cada lado = 140x140
    assert received_crop.shape == (140, 140, 3)


def test_frame_result_meets_latency_target_property() -> None:
    pipeline = SmartVisionPipeline(
        vehicle_detector=_FakeVehicleDetector([]),
        plate_detector=_FakePlateDetector([]),
        ocr_reader=_FakeOCRReader(None),
        vehicle_classifier=_FakeVehicleClassifier(),
    )
    result = pipeline.process_frame(_make_frame())
    # com componentes fake instantâneos, deve estar bem dentro da meta
    assert result.meets_latency_target is True


def test_from_config_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        SmartVisionPipeline.from_config(tmp_path / "nao_existe.yaml")
