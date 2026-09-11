"""Testes da Etapa 7 — utilitário compartilhado crop_with_margin."""

from __future__ import annotations

import numpy as np

from src.utils.geometry import crop_with_margin


def test_crop_without_margin_matches_exact_bbox() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = crop_with_margin(image, 50, 50, 150, 100, margin_ratio=0.0)
    assert crop.shape == (50, 100, 3)  # (y2-y1, x2-x1, canais)


def test_crop_with_margin_expands_region() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    crop_no_margin = crop_with_margin(image, 50, 50, 150, 100, margin_ratio=0.0)
    crop_with_margin_ = crop_with_margin(image, 50, 50, 150, 100, margin_ratio=0.2)

    assert crop_with_margin_.shape[0] > crop_no_margin.shape[0]
    assert crop_with_margin_.shape[1] > crop_no_margin.shape[1]


def test_crop_with_margin_clips_to_image_bounds() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # bbox já encosta nas bordas — margem não deve extrapolar a imagem
    crop = crop_with_margin(image, 0, 0, 100, 100, margin_ratio=0.5)
    assert crop.shape == (100, 100, 3)
