"""Testes da Etapa 4 — conversão do manifest para formato YOLO.

Usa imagens sintéticas geradas em `tmp_path` (não dependemos das imagens
reais do UFPR-VeSV, indisponíveis neste ambiente de desenvolvimento).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from src.preprocessing.yolo_converter import (
    ConversionReport,
    convert_manifest_to_yolo,
    write_data_yaml,
)
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
                "plate": "DEF5678",
                "plate_bbox_x1": 50,
                "plate_bbox_y1": 60,
                "plate_bbox_x2": 130,
                "plate_bbox_y2": 100,
                "split": "val",
            },
            {
                "filename": "img_c_missing.jpg",  # não terá arquivo correspondente em disco
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
    _write_dummy_image(d / "img_a.jpg", size=(400, 800))  # h=400, w=800
    _write_dummy_image(d / "img_b.jpg", size=(200, 400))  # h=200, w=400
    # img_c_missing.jpg propositalmente não é criada
    return d


def test_convert_manifest_to_yolo_creates_expected_structure(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    output_dir = tmp_path / "yolo_dataset"

    report = convert_manifest_to_yolo(manifest, images_dir=images_dir, output_dir=output_dir)

    assert isinstance(report, ConversionReport)
    assert report.total == 3
    assert report.converted == 2
    assert report.skipped_missing_image == 1
    assert report.skipped_unreadable_image == 0

    assert (output_dir / "images" / "train" / "img_a.jpg").exists()
    assert (output_dir / "labels" / "train" / "img_a.txt").exists()
    assert (output_dir / "images" / "val" / "img_b.jpg").exists()
    assert (output_dir / "labels" / "val" / "img_b.txt").exists()
    # o registro com imagem ausente não deve gerar arquivo nenhum
    assert not (output_dir / "images" / "test" / "img_c_missing.jpg").exists()


def test_convert_manifest_to_yolo_label_values_are_normalized_correctly(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    output_dir = tmp_path / "yolo_dataset"
    convert_manifest_to_yolo(manifest, images_dir=images_dir, output_dir=output_dir)

    label_content = (output_dir / "labels" / "train" / "img_a.txt").read_text().strip()
    class_id, xc, yc, w, h = label_content.split()

    # img_a.jpg tem w=800, h=400; bbox (100,100,200,150)
    assert class_id == "0"
    assert float(xc) == pytest.approx((100 + 200) / 2 / 800, abs=1e-4)
    assert float(yc) == pytest.approx((100 + 150) / 2 / 400, abs=1e-4)
    assert float(w) == pytest.approx((200 - 100) / 800, abs=1e-4)
    assert float(h) == pytest.approx((150 - 100) / 400, abs=1e-4)
    # todos os valores devem estar normalizados entre 0 e 1
    for value in (xc, yc, w, h):
        assert 0.0 <= float(value) <= 1.0


def test_convert_manifest_to_yolo_raises_without_split_column(tmp_path, images_dir) -> None:
    manifest = _make_manifest().drop(columns=["split"])
    with pytest.raises(DatasetError):
        convert_manifest_to_yolo(manifest, images_dir=images_dir, output_dir=tmp_path / "out")


def test_convert_manifest_to_yolo_raises_on_invalid_copy_mode(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    with pytest.raises(DatasetError):
        convert_manifest_to_yolo(
            manifest, images_dir=images_dir, output_dir=tmp_path / "out", copy_mode="mover"
        )


def test_convert_manifest_to_yolo_symlink_mode(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    output_dir = tmp_path / "yolo_dataset_symlink"
    report = convert_manifest_to_yolo(
        manifest, images_dir=images_dir, output_dir=output_dir, copy_mode="symlink"
    )
    assert report.converted == 2
    dst = output_dir / "images" / "train" / "img_a.jpg"
    assert dst.is_symlink()


def test_write_data_yaml_has_expected_fields(tmp_path) -> None:
    output_dir = tmp_path / "yolo_dataset"
    output_dir.mkdir()
    data_yaml_path = write_data_yaml(output_dir)

    assert data_yaml_path.exists()
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["nc"] == 1
    assert data["names"] == ["plate"]
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["test"] == "images/test"


def test_convert_manifest_to_yolo_also_writes_data_yaml(tmp_path, images_dir) -> None:
    manifest = _make_manifest()
    output_dir = tmp_path / "yolo_dataset"
    convert_manifest_to_yolo(manifest, images_dir=images_dir, output_dir=output_dir)
    assert (output_dir / "data.yaml").exists()
