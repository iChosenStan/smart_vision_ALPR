"""Detector de veículos usando um modelo YOLO genérico pré-treinado em COCO.

Este módulo NÃO faz fine-tuning — usa pesos pré-treinados diretamente,
pois o dataset do projeto (UFPR-VeSV) não fornece bounding box de veículo
(ver docs/02_dataset.md). Apenas as classes relevantes ao domínio de
veículos são mantidas na saída (car, motorcycle, bus, truck por padrão,
seguindo o mapeamento de 80 classes do COCO).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from src.detection.base_yolo_detector import BaseYOLODetector
from src.detection.schemas import VehicleDetection
from src.utils.exceptions import ModelLoadError, VehicleDetectionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/detection.yaml")

_DEFAULT_TARGET_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector(BaseYOLODetector):
    """Wrapper sobre `ultralytics.YOLO` especializado em detecção de veículos."""

    detection_cls = VehicleDetection

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        target_classes: Optional[Dict[int, str]] = None,
        _model: Optional[Any] = None,
    ) -> None:
        """
        Args:
            weights: Caminho ou nome do checkpoint Ultralytics (ex: "yolov8n.pt").
                Se for apenas o nome (sem caminho), o Ultralytics baixa
                automaticamente na primeira execução e faz cache local.
            device: "cuda" ou "cpu". Faz fallback automático para CPU se CUDA
                não estiver disponível no ambiente.
            confidence_threshold: Confiança mínima para manter uma detecção.
            iou_threshold: Threshold de IoU usado no NMS interno do modelo.
            image_size: Tamanho de entrada da imagem (lado, em pixels).
            target_classes: Mapeamento {class_id: nome} das classes COCO
                relevantes. Se None, usa o padrão de veículos.
            _model: Instância de modelo já carregada — usado para injeção de
                dependência em testes (evita download real de pesos).
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

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "VehicleDetector":
        """Instancia o detector a partir de `configs/detection.yaml`."""
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

    def detect(self, image: np.ndarray) -> List[VehicleDetection]:
        """Detecta veículos em uma imagem.

        Args:
            image: Array de imagem (H, W, 3), no formato aceito pelo
                `ultralytics.YOLO.predict` (BGR do OpenCV funciona).

        Returns:
            Lista de `VehicleDetection`, já filtrada para conter apenas as
            classes definidas em `target_classes`.

        Raises:
            VehicleDetectionError: Se a inferência falhar.
        """
        try:
            result = self._predict_raw(image)
        except Exception as exc:  # noqa: BLE001
            raise VehicleDetectionError(f"Falha na inferência de detecção: {exc}") from exc

        return self._parse_results(result)
