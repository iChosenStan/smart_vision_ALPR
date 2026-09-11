"""Testes da Etapa 6 — VehicleClassificationDataset (imagens sintéticas)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.classification.dataset import VehicleClassificationDataset
from src.classification.label_encoder import LabelEncoders
from src.classification.transforms import build_eval_transform
from src.utils.exceptions import DatasetError


def _make_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"filename": "img_a.jpg", "type": "car", "make": "fiat", "model": "uno", "color": "white"},
            {"filename": "img_b.jpg", "type": "truck", "make": "ford", "model": "f250", "color": "red"},
        ]
    )


@pytest.fixture
def images_dir(tmp_path) -> Path:
    d = tmp_path / "images"
    d.mkdir()
    cv2.imwrite(str(d / "img_a.jpg"), np.zeros((300, 400, 3), dtype=np.uint8))
    cv2.imwrite(str(d / "img_b.jpg"), np.full((300, 400, 3), 200, dtype=np.uint8))
    return d


def test_dataset_returns_correct_item_structure(images_dir) -> None:
    manifest = _make_manifest()
    encoders = LabelEncoders.fit(manifest)
    dataset = VehicleClassificationDataset(
        manifest, images_dir, encoders, transform=build_eval_transform(224)
    )

    item = dataset[0]

    assert item["image"].shape == (3, 224, 224)
    assert isinstance(item["type_label"], torch.Tensor)
    assert item["type_label"].dtype == torch.long


def test_dataset_labels_match_encoder(images_dir) -> None:
    manifest = _make_manifest()
    encoders = LabelEncoders.fit(manifest)
    dataset = VehicleClassificationDataset(
        manifest, images_dir, encoders, transform=build_eval_transform(224)
    )

    item = dataset[1]  # img_b.jpg -> truck/ford/f250/red
    assert item["type_label"].item() == encoders.encode("type", "truck")
    assert item["make_label"].item() == encoders.encode("make", "ford")


def test_dataset_len_matches_manifest(images_dir) -> None:
    manifest = _make_manifest()
    encoders = LabelEncoders.fit(manifest)
    dataset = VehicleClassificationDataset(
        manifest, images_dir, encoders, transform=build_eval_transform(224)
    )
    assert len(dataset) == 2


def test_dataset_raises_on_missing_task_columns(images_dir) -> None:
    manifest = _make_manifest().drop(columns=["color"])
    encoders = LabelEncoders.fit(_make_manifest())
    with pytest.raises(DatasetError):
        VehicleClassificationDataset(manifest, images_dir, encoders, transform=build_eval_transform(224))


def test_dataset_raises_when_image_file_missing(images_dir) -> None:
    manifest = pd.DataFrame(
        [{"filename": "nao_existe.jpg", "type": "car", "make": "fiat", "model": "uno", "color": "white"}]
    )
    encoders = LabelEncoders.fit(_make_manifest())
    dataset = VehicleClassificationDataset(manifest, images_dir, encoders, transform=build_eval_transform(224))
    with pytest.raises(DatasetError):
        _ = dataset[0]
