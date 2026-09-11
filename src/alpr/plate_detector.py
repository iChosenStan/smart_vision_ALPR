"""Detector de placa usando um modelo YOLO fine-tunado no UFPR-VeSV.

Diferente do `VehicleDetector` (Etapa 3), este modelo é treinado
especificamente para o dataset do projeto (ver
`src/training/train_plate_detector.py`), com uma única classe: "plate".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from src.alpr.schemas import PlateDetection
from src.detection.base_yolo_detector import BaseYOLODetector
from src.utils.exceptions import ModelLoadError, PlateDetectionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/plate_detection.yaml")

_DEFAULT_TARGET_CLASSES = {0: "plate"}


class PlateDetector(BaseYOLODetector):
    """Wrapper sobre `ultralytics.YOLO` especializado em detecção de placa."""

    detection_cls = PlateDetection

    def __init__(
        self,
        weights: str = "models/checkpoints/plate_detector_best.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        target_classes: Optional[Dict[int, str]] = None,
        _model: Optional[Any] = None,
    ) -> None:
        """
        Args:
            weights: Caminho para os pesos fine-tunados localmente (não é
                baixado automaticamente — precisa existir em disco, gerado
                por `src/training/train_plate_detector.py`).
            device: "cuda" ou "cpu". Faz fallback automático para CPU.
            confidence_threshold: Confiança mínima para manter uma detecção.
            iou_threshold: Threshold de IoU usado no NMS interno do modelo.
            image_size: Tamanho de entrada da imagem (lado, em pixels).
            target_classes: Mapeamento {class_id: nome}. Modelo de classe
                única por padrão ({0: "plate"}).
            _model: Instância de modelo já carregada — usado para injeção de
                dependência em testes.
        """
        super().__init__(
            weights=weights,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            target_classes=target_classes or dict(_DEFAULT_TARGET_CLASSES),
            _model=_model,
        )

    @staticmethod
    def _load_model(weights: str, device: str) -> Any:
        weights_path = Path(weights)
        if not weights_path.exists():
            raise ModelLoadError(
                f"Pesos do detector de placa não encontrados em '{weights}'. "
                "Rode o treino primeiro: python scripts/train_plate_detector.py"
            )
        return BaseYOLODetector._load_model(weights, device)

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "PlateDetector":
        """Instancia o detector a partir de `configs/plate_detection.yaml`."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ModelLoadError(f"Arquivo de configuração não encontrado: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls(
            weights=config["model"]["weights"],
            device=config["model"]["device"],
            confidence_threshold=config["inference"]["confidence_threshold"],
            iou_threshold=config["inference"]["iou_threshold"],
            image_size=config["inference"]["image_size"],
            target_classes={int(k): v for k, v in config["target_classes"].items()},
        )

    def detect(self, image: np.ndarray) -> List[PlateDetection]:
        """Detecta placa(s) em uma imagem (tipicamente o recorte de um veículo).

        Args:
            image: Array de imagem (H, W, 3), formato BGR do OpenCV.

        Returns:
            Lista de `PlateDetection` (geralmente 0 ou 1 por veículo).

        Raises:
            PlateDetectionError: Se a inferência falhar.
        """
        try:
            result = self._predict_raw(image)
        except Exception as exc:  # noqa: BLE001
            raise PlateDetectionError(f"Falha na inferência de detecção de placa: {exc}") from exc

        return self._parse_results(result)

    @staticmethod
    def crop_best_plate(image: np.ndarray, detections: List[PlateDetection]) -> Optional[np.ndarray]:
        """Retorna o recorte da placa de maior confiança, ou None se não houver detecções.

        Útil para alimentar diretamente o estágio de OCR (Etapa 5).
        """
        if not detections:
            return None

        best = max(detections, key=lambda d: d.confidence)
        x1, y1, x2, y2 = (int(v) for v in best.bbox_xyxy)
        return image[y1:y2, x1:x2]
