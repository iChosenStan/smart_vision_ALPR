"""Codificação de rótulos categóricos (type/make/model/color) para índices
inteiros, usada tanto no treino quanto na inferência do classificador
multi-tarefa (Etapa 6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TASKS = ("type", "make", "model", "color")


@dataclass
class LabelEncoders:
    """Mapeamentos classe↔índice para cada uma das 4 tarefas de classificação."""

    class_to_idx: Dict[str, Dict[str, int]]
    idx_to_class: Dict[str, Dict[int, str]]

    @property
    def num_classes(self) -> Dict[str, int]:
        return {task: len(mapping) for task, mapping in self.class_to_idx.items()}

    def encode(self, task: str, label: str) -> int:
        try:
            return self.class_to_idx[task][label]
        except KeyError as exc:
            raise DatasetError(f"Classe desconhecida para '{task}': {label!r}") from exc

    def decode(self, task: str, idx: int) -> str:
        try:
            return self.idx_to_class[task][idx]
        except KeyError as exc:
            raise DatasetError(f"Índice desconhecido para '{task}': {idx!r}") from exc

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.class_to_idx, f, ensure_ascii=False, indent=2, sort_keys=True)
        logger.info(f"Label encoders salvos em {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LabelEncoders":
        path = Path(path)
        if not path.exists():
            raise DatasetError(f"Arquivo de label encoders não encontrado: {path}")
        with open(path, "r", encoding="utf-8") as f:
            class_to_idx: Dict[str, Dict[str, int]] = json.load(f)
        idx_to_class = {
            task: {idx: label for label, idx in mapping.items()} for task, mapping in class_to_idx.items()
        }
        return cls(class_to_idx=class_to_idx, idx_to_class=idx_to_class)

    @classmethod
    def fit(cls, manifest: pd.DataFrame, tasks: Optional[List[str]] = None) -> "LabelEncoders":
        """Constrói os encoders a partir dos valores únicos do manifest.

        Deve ser chamado apenas com o split de TREINO, para evitar que
        classes vistas somente em val/test vazem para o vocabulário do
        modelo antes da hora.
        """
        tasks = tasks or list(_TASKS)
        class_to_idx: Dict[str, Dict[str, int]] = {}
        for task in tasks:
            if task not in manifest.columns:
                raise DatasetError(f"Coluna '{task}' não encontrada no manifest")
            unique_values = sorted(manifest[task].astype(str).unique())
            class_to_idx[task] = {label: idx for idx, label in enumerate(unique_values)}

        idx_to_class = {
            task: {idx: label for label, idx in mapping.items()} for task, mapping in class_to_idx.items()
        }
        logger.info(f"Label encoders ajustados: {[(t, len(m)) for t, m in class_to_idx.items()]}")
        return cls(class_to_idx=class_to_idx, idx_to_class=idx_to_class)
