"""Classificador de veículo (tipo, marca, modelo, cor) — modelo
multi-tarefa (ResNet34 + 4 cabeças) treinado no UFPR-VeSV (Etapa 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.classification.label_encoder import LabelEncoders
from src.classification.multi_task_model import VehicleMultiTaskNet
from src.classification.schemas import AttributePrediction, VehicleClassification
from src.classification.transforms import build_eval_transform
from src.utils.exceptions import ClassificationError, ModelLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/classification.yaml")
_TASKS = ("type", "make", "model", "color")


class VehicleClassifier:
    """Wrapper de inferência sobre `VehicleMultiTaskNet`."""

    def __init__(
        self,
        weights: str = "models/checkpoints/vehicle_classifier_best.pt",
        encoders_path: str = "models/checkpoints/vehicle_classifier_encoders.json",
        device: str = "cuda",
        image_size: int = 224,
        _model: Optional[Any] = None,
        _encoders: Optional[LabelEncoders] = None,
    ) -> None:
        """
        Args:
            weights: Caminho para os pesos fine-tunados (`state_dict` do
                PyTorch), gerados por `scripts/train_vehicle_classifier.py`.
            encoders_path: Caminho para o JSON de label encoders (classe↔índice),
                salvo junto com os pesos ao final do treino.
            device: "cuda" ou "cpu". Faz fallback automático para CPU.
            image_size: Tamanho da imagem de entrada esperado pelo modelo.
            _model: Instância de modelo já carregada — injeção de dependência
                para testes.
            _encoders: `LabelEncoders` já carregado — injeção de dependência
                para testes.
        """
        self.weights = weights
        self.image_size = image_size
        self.device = self._resolve_device(device)
        self.encoders = _encoders if _encoders is not None else LabelEncoders.load(encoders_path)
        self._model = (
            _model if _model is not None else self._load_model(weights, self.encoders, self.device)
        )
        self._transform = build_eval_transform(image_size)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "cuda":
            return device
        if not torch.cuda.is_available():
            logger.warning("CUDA solicitado mas não disponível — usando CPU.")
            return "cpu"
        return device

    @staticmethod
    def _load_model(weights: str, encoders: LabelEncoders, device: str) -> Any:
        weights_path = Path(weights)
        if not weights_path.exists():
            raise ModelLoadError(
                f"Pesos do classificador não encontrados em '{weights}'. "
                "Rode o treino primeiro: python scripts/train_vehicle_classifier.py"
            )

        model = VehicleMultiTaskNet(num_classes=encoders.num_classes, pretrained=False)
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Falha ao carregar pesos '{weights}': {exc}") from exc

        model.to(device)
        model.eval()
        logger.info(f"Classificador de veículo carregado: {weights} (device={device})")
        return model

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "VehicleClassifier":
        """Instancia o classificador a partir de `configs/classification.yaml`."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ModelLoadError(f"Arquivo de configuração não encontrado: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls(
            weights=config["model"]["weights"],
            encoders_path=config["model"]["encoders_path"],
            device=config["model"]["device"],
            image_size=config["inference"]["image_size"],
        )

    def classify(self, image: np.ndarray) -> VehicleClassification:
        """Classifica type/make/model/color a partir de uma imagem (BGR, OpenCV).

        Args:
            image: Imagem do veículo — tipicamente o frame completo, já
                que o dataset não fornece bbox de veículo (ver
                docs/06_classification.md).

        Returns:
            `VehicleClassification` com a predição de maior probabilidade
            (e sua confiança) para cada um dos 4 atributos.

        Raises:
            ClassificationError: Se a inferência falhar.
        """
        try:
            image_rgb = image[:, :, ::-1]  # BGR -> RGB
            tensor = self._transform(image_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self._model(tensor)
        except Exception as exc:  # noqa: BLE001
            raise ClassificationError(f"Falha na inferência de classificação: {exc}") from exc

        predictions = {}
        for task in _TASKS:
            probs = F.softmax(outputs[task][0], dim=0)
            top_idx = int(torch.argmax(probs).item())
            predictions[task] = AttributePrediction(
                label=self.encoders.decode(task, top_idx),
                confidence=float(probs[top_idx].item()),
            )

        return VehicleClassification(
            type=predictions["type"],
            make=predictions["make"],
            model=predictions["model"],
            color=predictions["color"],
        )
