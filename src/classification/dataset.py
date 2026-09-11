"""Dataset PyTorch para o classificador multi-tarefa de veículo (Etapa 6).

Diferente do que seria ideal, usamos a IMAGEM COMPLETA como entrada — o
UFPR-VeSV não fornece bounding box de veículo (ver docs/02_dataset.md e
docs/03_vehicle_detection.md), e cada imagem do dataset já enquadra um
único veículo principal, então treinar sobre o frame inteiro é a
abordagem viável disponível.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Union

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.classification.label_encoder import LabelEncoders
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TASKS = ("type", "make", "model", "color")


class VehicleClassificationDataset(Dataset):
    """Dataset de classificação multi-tarefa (type, make, model, color)."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        images_dir: Union[str, Path],
        encoders: LabelEncoders,
        transform: Callable,
    ) -> None:
        """
        Args:
            manifest: DataFrame já filtrado para o split desejado (ex:
                `manifest[manifest["split"] == "train"]`).
            images_dir: Diretório com as imagens originais.
            encoders: `LabelEncoders` ajustado (tipicamente no split de treino).
            transform: Callable `(np.ndarray RGB) -> torch.Tensor` — ver
                `src.classification.transforms`.
        """
        missing_cols = set(_TASKS) - set(manifest.columns)
        if missing_cols:
            raise DatasetError(f"Manifest não possui as colunas: {missing_cols}")

        self.manifest = manifest.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.encoders = encoders
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        image_path = self.images_dir / row["filename"]

        image = cv2.imread(str(image_path))
        if image is None:
            raise DatasetError(f"Não foi possível ler a imagem: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image)

        item: Dict[str, torch.Tensor] = {"image": image_tensor}
        for task in _TASKS:
            label_idx = self.encoders.encode(task, str(row[task]))
            item[f"{task}_label"] = torch.tensor(label_idx, dtype=torch.long)

        return item
