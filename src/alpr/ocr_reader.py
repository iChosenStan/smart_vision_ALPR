"""Reconhecimento de texto (OCR) da placa, usando `paddleocr.TextRecognition`
com um modelo fine-tunado no UFPR-VeSV.

Diferente do pipeline completo do PaddleOCR (detecção + reconhecimento),
usamos apenas o módulo de reconhecimento — a Etapa 4 já entrega um
recorte apertado da placa, então a detecção de texto seria redundante.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml

from src.alpr.plate_format import matches_known_plate_format, normalize_plate_text
from src.alpr.schemas import OCRResult
from src.utils.exceptions import ModelLoadError, OCRError
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/plate_recognition.yaml")


class OCRReader:
    """Wrapper sobre `paddleocr.TextRecognition` especializado em placas."""

    def __init__(
        self,
        model_dir: Optional[str] = "models/checkpoints/plate_recognizer",
        model_name: str = "PP-OCRv5_mobile_rec",
        device: str = "gpu",
        _model: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model_dir: Diretório com os pesos fine-tunados exportados
                (formato de inferência do PaddleX/PaddleOCR). Se None,
                usa o modelo genérico pré-treinado `model_name` (sem
                fine-tuning) — útil para comparação/fallback.
            model_name: Nome do modelo base (usado para baixar pesos
                genéricos caso `model_dir` seja None, ou como referência
                de arquitetura).
            device: "gpu" ou "cpu".
            _model: Instância de modelo já carregada — usado para injeção
                de dependência em testes (evita download real de pesos).
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.device = device
        self._model = _model if _model is not None else self._load_model(model_dir, model_name, device)

    @staticmethod
    def _load_model(model_dir: Optional[str], model_name: str, device: str) -> Any:
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise ModelLoadError(
                "Pacote 'paddleocr' não instalado. Rode: pip install paddleocr paddlepaddle"
            ) from exc

        if model_dir is not None:
            model_dir_path = Path(model_dir)
            if not model_dir_path.exists():
                raise ModelLoadError(
                    f"Pesos do reconhecedor de placa não encontrados em '{model_dir}'. "
                    "Rode o treino primeiro: python scripts/train_plate_recognizer.py"
                )

        try:
            model = TextRecognition(model_name=model_name, model_dir=model_dir, device=device)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Falha ao carregar modelo de OCR: {exc}") from exc

        logger.info(f"Modelo de OCR carregado: {model_name} (model_dir={model_dir}, device={device})")
        return model

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "OCRReader":
        """Instancia o leitor a partir de `configs/plate_recognition.yaml`."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ModelLoadError(f"Arquivo de configuração não encontrado: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls(
            model_dir=config["model"]["model_dir"],
            model_name=config["model"]["model_name"],
            device=config["model"]["device"],
        )

    def read(self, plate_image: np.ndarray) -> Optional[OCRResult]:
        """Reconhece o texto em um recorte de placa.

        Args:
            plate_image: Array de imagem (H, W, 3) — tipicamente a saída de
                `PlateDetector.crop_best_plate()`.

        Returns:
            `OCRResult`, ou None se o recorte for vazio/inválido.

        Raises:
            OCRError: Se a inferência falhar.
        """
        if plate_image is None or plate_image.size == 0:
            return None

        try:
            outputs = list(self._model.predict(input=plate_image, batch_size=1))
        except Exception as exc:  # noqa: BLE001
            raise OCRError(f"Falha na inferência de OCR: {exc}") from exc

        if not outputs:
            return None

        res = outputs[0]
        raw_text = res["rec_text"]
        confidence = float(res["rec_score"])
        normalized = normalize_plate_text(raw_text)

        return OCRResult(
            text=raw_text,
            confidence=confidence,
            normalized_text=normalized,
            matches_known_format=matches_known_plate_format(normalized),
        )
