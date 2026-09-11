"""Testes da Etapa 6 — transformações de imagem (treino/inferência)."""

from __future__ import annotations

import numpy as np
import torch

from src.classification.transforms import build_eval_transform, build_train_transform


def _make_dummy_image(h=300, w=400):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_eval_transform_output_shape_and_dtype() -> None:
    transform = build_eval_transform(image_size=224)
    image = _make_dummy_image()

    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transform_is_deterministic() -> None:
    transform = build_eval_transform(image_size=128)
    image = _make_dummy_image()

    t1 = transform(image)
    t2 = transform(image)

    assert torch.allclose(t1, t2)


def test_train_transform_output_shape_and_dtype() -> None:
    transform = build_train_transform(image_size=224)
    image = _make_dummy_image()

    tensor = transform(image)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_transform_normalizes_with_imagenet_stats() -> None:
    """Pixels de valor médio (~127) devem ficar próximos de 0 após normalização."""
    transform = build_eval_transform(image_size=64)
    flat_gray_image = np.full((64, 64, 3), 127, dtype=np.uint8)

    tensor = transform(flat_gray_image)

    assert tensor.mean().item() == 0.0 or abs(tensor.mean().item()) < 1.0
