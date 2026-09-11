"""Testes da Etapa 5 — conversão do manifest para dataset de reconhecimento (OCR)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.rec_converter import RecConversionReport, convert_manifest_to_rec_dataset
from src.utils.exceptions import DatasetError


def _make_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "filename": "img_a.jpg",
                "plate": "ABC1234",
                "plate_bbox_x1": 100,
                "plate_bbox_y1": 100,
                "plate_bbox_x2": 200,
                "plate_bbox_y2": 150,
                "split": "train",
            },
            {
                "filename": "img_b.jpg",
                "plate": "DEF5G67",
                "plate_bbox_x1": 50,
                "plate_bbox_y1": 60,
                "plate_bbox_x2": 130,
                "plate_bbox_y2": 100,
                "split": "val",
            },
            {
                "filename": "img_c_missing.jpg",
                "plate": "GHI9999",
                "plate_bbox_x1": 10,
                "plate_bbox_y1": 10,
                "plate_bbox_x2": 40,
                "plate_bbox_y2": 40,
                "split": "test",
            },
        ]
    )


def _write_dummy_image(path: Path, size=(480, 640)) -> None:
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


@pytest.fixture
def images_dir(tmp_path) -> Path:
    d = tmp_path / "raw_images"
    d.mkdir()
    _write_dummy_image(d / "img_a.jpg", size=(400, 800))
    _write_dummy_image(d / "img_b.jpg", size=(200, 400))
    return d


def test_convert_manifest_to_rec_dataset_creates_expected_files(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    output_dir = tmp_path / "rec_dataset"

    report = convert_manifest_to_rec_dataset(manifest, images_dir=images_dir, output_dir=output_dir)

    assert isinstance(report, RecConversionReport)
    assert report.total == 3
    assert report.converted == 2
    assert report.skipped_missing_image == 1

    assert (output_dir / "images" / "train" / "img_a.jpg").exists()
    assert (output_dir / "images" / "val" / "img_b.jpg").exists()
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "val.txt").exists()
    assert (output_dir / "test.txt").exists()  # deve existir mesmo vazio


def test_convert_manifest_to_rec_dataset_label_format_is_tab_separated(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    output_dir = tmp_path / "rec_dataset"
    convert_manifest_to_rec_dataset(manifest, images_dir=images_dir, output_dir=output_dir)

    train_content = (output_dir / "train.txt").read_text(encoding="utf-8").strip()
    parts = train_content.split("\t")
    assert len(parts) == 2
    image_path, label = parts
    assert image_path == "images/train/img_a.jpg"
    assert label == "ABC1234"


def test_convert_manifest_to_rec_dataset_crop_is_smaller_than_original(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    output_dir = tmp_path / "rec_dataset"
    convert_manifest_to_rec_dataset(manifest, images_dir=images_dir, output_dir=output_dir, margin_ratio=0.0)

    original = cv2.imread(str(images_dir / "img_a.jpg"))
    crop = cv2.imread(str(output_dir / "images" / "train" / "img_a.jpg"))

    assert crop.shape[0] < original.shape[0]
    assert crop.shape[1] < original.shape[1]
    # bbox (100,100,200,150) sem margem -> largura 100, altura 50
    assert crop.shape[1] == 100
    assert crop.shape[0] == 50


def test_convert_manifest_to_rec_dataset_applies_margin(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    output_dir_no_margin = tmp_path / "rec_no_margin"
    output_dir_with_margin = tmp_path / "rec_with_margin"

    convert_manifest_to_rec_dataset(manifest, images_dir=images_dir, output_dir=output_dir_no_margin, margin_ratio=0.0)
    convert_manifest_to_rec_dataset(manifest, images_dir=images_dir, output_dir=output_dir_with_margin, margin_ratio=0.2)

    crop_no_margin = cv2.imread(str(output_dir_no_margin / "images" / "train" / "img_a.jpg"))
    crop_with_margin = cv2.imread(str(output_dir_with_margin / "images" / "train" / "img_a.jpg"))

    assert crop_with_margin.shape[0] > crop_no_margin.shape[0]
    assert crop_with_margin.shape[1] > crop_no_margin.shape[1]


def test_convert_manifest_to_rec_dataset_raises_without_split_column(tmp_path, images_dir) -> None:
    manifest = _make_manifest().drop(columns=["split"])
    with pytest.raises(DatasetError):
        convert_manifest_to_rec_dataset(manifest, images_dir=images_dir, output_dir=tmp_path / "out")


def test_convert_manifest_to_rec_dataset_generates_char_dict(tmp_path, images_dir) -> None:
    """Regressão: PaddleOCR exige dict.txt em {dataset_dir}/dict.txt para
    treinar — bug real descoberto em execução prática (Windows/Colab).
    """
    manifest = _make_manifest()
    output_dir = tmp_path / "rec_dataset"
    convert_manifest_to_rec_dataset(manifest, images_dir=images_dir, output_dir=output_dir)

    dict_path = output_dir / "dict.txt"
    assert dict_path.exists()

    chars = dict_path.read_text(encoding="utf-8").splitlines()
    # placas do manifest: "ABC1234", "DEF5G67", "GHI9999" (ausente na imagem)
    # -> caracteres únicos esperados: A,B,C,D,E,F,G,H,I,1,2,3,4,5,6,7,9
    assert set(chars) == set("ABCDEFGHI1234567 9".replace(" ", ""))
    assert len(chars) == len(set(chars))  # sem duplicatas
    assert chars == sorted(chars)  # ordem determinística
