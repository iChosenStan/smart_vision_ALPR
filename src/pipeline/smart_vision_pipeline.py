"""Pipeline de inferência ponta a ponta do SmartVision ALPR (Etapa 7).

Integra os 4 módulos construídos nas Etapas 3-6:

    VehicleDetector → [recorte] → PlateDetector → [recorte] → OCRReader
                                └────────────────→ VehicleClassifier

Decisão registrada (ver docs/07_pipeline.md): tanto o `PlateDetector`
quanto o `VehicleClassifier` recebem o RECORTE do veículo detectado (não
o frame inteiro), permitindo suportar múltiplos veículos por frame.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from src.alpr.ocr_reader import OCRReader
from src.alpr.plate_detector import PlateDetector
from src.classification.vehicle_classifier import VehicleClassifier
from src.detection.schemas import VehicleDetection
from src.detection.vehicle_detector import VehicleDetector
from src.pipeline.schemas import FrameResult, VehicleResult
from src.utils.geometry import crop_with_margin
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/pipeline.yaml")


class SmartVisionPipeline:
    """Orquestra os 4 módulos do pipeline ALPR ponta a ponta, com medição
    de latência por estágio.
    """

    def __init__(
        self,
        vehicle_detector: VehicleDetector,
        plate_detector: PlateDetector,
        ocr_reader: OCRReader,
        vehicle_classifier: VehicleClassifier,
        crop_margin_ratio: float = 0.10,
        min_vehicle_area: float = 0.0,
    ) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader
        self.vehicle_classifier = vehicle_classifier
        self.crop_margin_ratio = crop_margin_ratio
        self.min_vehicle_area = min_vehicle_area

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "SmartVisionPipeline":
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls(
            vehicle_detector=VehicleDetector.from_config(),
            plate_detector=PlateDetector.from_config(),
            ocr_reader=OCRReader.from_config(),
            vehicle_classifier=VehicleClassifier.from_config(),
            crop_margin_ratio=config["pipeline"]["crop_margin_ratio"],
            min_vehicle_area=config["pipeline"]["min_vehicle_area"],
        )

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        vehicle_detections = self.vehicle_detector.detect(frame)
        vehicle_detection_ms = (time.perf_counter() - t0) * 1000

        vehicle_results = [
            self._process_vehicle(frame, detection)
            for detection in vehicle_detections
            if detection.area >= self.min_vehicle_area
        ]

        total_ms = (time.perf_counter() - t_start) * 1000
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        logger.info(
            f"Frame processado: {len(vehicle_results)} veículo(s) — "
            f"{total_ms:.1f}ms total ({fps:.1f} FPS)"
        )

        return FrameResult(
            vehicles=vehicle_results,
            vehicle_detection_ms=vehicle_detection_ms,
            total_latency_ms=total_ms,
            fps=fps,
        )

    def _process_vehicle(self, frame: np.ndarray, detection: VehicleDetection) -> VehicleResult:
        x1, y1, x2, y2 = (int(v) for v in detection.bbox_xyxy)
        vehicle_crop = crop_with_margin(frame, x1, y1, x2, y2, self.crop_margin_ratio)

        t0 = time.perf_counter()
        plate_detections = self.plate_detector.detect(vehicle_crop)
        plate_detection_ms = (time.perf_counter() - t0) * 1000

        plate_crop = PlateDetector.crop_best_plate(vehicle_crop, plate_detections)
        best_plate = max(plate_detections, key=lambda d: d.confidence) if plate_detections else None

        ocr_result = None
        ocr_ms = 0.0
        if plate_crop is not None:
            t0 = time.perf_counter()
            ocr_result = self.ocr_reader.read(plate_crop)
            ocr_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        classification = self.vehicle_classifier.classify(vehicle_crop)
        classification_ms = (time.perf_counter() - t0) * 1000

        return VehicleResult(
            vehicle_detection=detection,
            plate_detection=best_plate,
            ocr_result=ocr_result,
            classification=classification,
            plate_detection_ms=plate_detection_ms,
            ocr_ms=ocr_ms,
            classification_ms=classification_ms,
            vehicle_crop=vehicle_crop,
            plate_crop=plate_crop,
        )
