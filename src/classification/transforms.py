"""Transformações de imagem para treino/inferência do classificador de
veículo (Etapa 6). Compartilhado entre `VehicleClassificationDataset`
(treino) e `VehicleClassifier` (inferência) para garantir que o
pré-processamento seja idêntico nos dois contextos.

IMPORTANTE: as transforms são classes com `__call__` (não closures/funções
aninhadas). Descoberto na prática (Windows): `DataLoader` com
`num_workers > 0` usa o método `spawn` no Windows (em vez de `fork`, usado
no Linux), que exige que os objetos passados aos processos filhos sejam
serializáveis via `pickle` — funções aninhadas (`def transform(...)`
dentro de outra função) não são picklable, e o treino falha com
`AttributeError: Can't pickle local object ...`. Classes de nível de
módulo resolvem isso.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    image = cv2.resize(image_rgb, (image_size, image_size))
    image = image.astype(np.float32) / 255.0
    image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(image).permute(2, 0, 1).float()


class _EvalTransform:
    """Resize + normalização ImageNet, sem augmentation (determinístico)."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image_rgb: np.ndarray) -> torch.Tensor:
        return _to_tensor(image_rgb, self.image_size)


class _TrainTransform:
    """Resize + flip horizontal aleatório (p=0.5) + normalização ImageNet."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image_rgb: np.ndarray) -> torch.Tensor:
        if random.random() < 0.5:
            image_rgb = np.ascontiguousarray(image_rgb[:, ::-1, :])
        return _to_tensor(image_rgb, self.image_size)


def build_train_transform(image_size: int = 224) -> _TrainTransform:
    """Transform de treino: resize + flip horizontal aleatório (p=0.5) +
    normalização ImageNet (compatível com o backbone pré-treinado).
    """
    return _TrainTransform(image_size)


def build_eval_transform(image_size: int = 224) -> _EvalTransform:
    """Transform de avaliação/inferência: resize + normalização ImageNet,
    sem augmentation (determinístico).
    """
    return _EvalTransform(image_size)
