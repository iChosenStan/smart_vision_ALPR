"""Classe base para wrappers de detecção baseados em `ultralytics.YOLO`.

Compartilhada por `VehicleDetector` (Etapa 3 — modelo genérico COCO, sem
fine-tuning) e `PlateDetector` (Etapa 4 — modelo fine-tunado no UFPR-VeSV),
concentrando a lógica comum de carregamento de modelo, resolução de
device e parsing/filtragem de detecções.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.exceptions import ModelLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseYOLODetector:
    """Wrapper genérico sobre `ultralytics.YOLO`.

    Subclasses definem apenas os defaults específicos do domínio (pesos,
    classes-alvo) e, opcionalmente, o tipo de dataclass de retorno via
    `_build_detection()`.
    """

    #: Dataclass usado para representar cada detecção — subclasses devem sobrescrever.
    detection_cls: type

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        target_classes: Optional[Dict[int, str]] = None,
        _model: Optional[Any] = None,
    ) -> None:
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.target_classes = target_classes or {}
        self.device = self._resolve_device(device)
        self._model = _model if _model is not None else self._load_model(self.weights, self.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Faz fallback para CPU se CUDA for solicitado mas não estiver disponível."""
        if device != "cuda":
            return device
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA solicitado mas não disponível — usando CPU.")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch não encontrado ao checar CUDA — usando CPU.")
            return "cpu"
        return device

    @staticmethod
    def _load_model(weights: str, device: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "Pacote 'ultralytics' não instalado. Rode: pip install ultralytics"
            ) from exc

        try:
            model = YOLO(weights)
        except Exception as exc:  # noqa: BLE001 - qualquer falha de carregamento vira ModelLoadError
            raise ModelLoadError(f"Falha ao carregar pesos '{weights}': {exc}") from exc

        logger.info(f"Modelo carregado: {weights} (device={device})")
        return model

    def _predict_raw(self, image: np.ndarray) -> Any:
        """Roda a inferência bruta do Ultralytics e retorna o primeiro `Results`."""
        results = self._model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        return results[0]

    def _parse_results(self, result: Any) -> List[Any]:
        """Converte a saída bruta do Ultralytics em instâncias de `detection_cls`,
        filtrando para conter apenas as classes definidas em `target_classes`.
        """
        detections: List[Any] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            if class_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            detections.append(
                self.detection_cls(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(boxes.conf[i]),
                    class_id=class_id,
                    class_name=self.target_classes[class_id],
                )
            )

        return detections
