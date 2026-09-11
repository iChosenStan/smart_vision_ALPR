"""Utilitários geométricos compartilhados."""
from __future__ import annotations

import numpy as np


def crop_with_margin(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int, margin_ratio: float = 0.0
) -> np.ndarray:
    h, w = image.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    mx, my = int(box_w * margin_ratio), int(box_h * margin_ratio)

    x1c = max(0, int(x1) - mx)
    y1c = max(0, int(y1) - my)
    x2c = min(w, int(x2) + mx)
    y2c = min(h, int(y2) + my)

    return image[y1c:y2c, x1c:x2c]
