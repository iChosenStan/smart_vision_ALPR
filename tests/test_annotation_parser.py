"""Testes da Etapa 2 — parsing de anotações e registro do dataset.

Usa `tests/fixtures/sample_annotations.json`, um subconjunto real
(19 registros) extraído do `annotations.json` original do UFPR-VeSV,
incluindo casos de placas duplicadas (mesmo veículo em múltiplas imagens).
"""

from pathlib import Path

import pytest

from src.preprocessing.annotation_parser import ParseReport, load_annotations
from src.preprocessing.dataset_registry import (
    build_manifest,
    split_by_vehicle,
    validate_images_exist,
)
from src.preprocessing.schemas import PlateAnnotation
from src.utils.exceptions import DatasetError

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_annotations.json"


@pytest.fixture(scope="module")
def loaded_annotations():
    annotations, report = load_annotations(FIXTURE_PATH)
    return annotations, report


def test_load_annotations_parses_all_valid_records(loaded_annotations) -> None:
    annotations, report = loaded_annotations
    assert isinstance(report, ParseReport)
    assert report.total == 19
    assert report.invalid == 0
    assert report.valid == 19
    assert len(annotations) == 19


def test_annotation_fields_are_normalized(loaded_annotations) -> None:
    annotations, _ = loaded_annotations
    first = annotations[0]
    assert isinstance(first, PlateAnnotation)
    assert first.filename == "img_00001.jpg"
    assert first.make == "chevrolet"
    assert first.plate == "MAV6C14"
    # infrared/rear_view devem virar bool, não string
    assert isinstance(first.infrared, bool)
    assert isinstance(first.rear_view, bool)


def test_bbox_xyxy_is_derived_correctly_from_corners(loaded_annotations) -> None:
    annotations, _ = loaded_annotations
    first = annotations[0]
    x1, y1, x2, y2 = first.bbox_xyxy
    # corners originais: (445,290) (597,281) (598,329) (445,338)
    assert (x1, y1, x2, y2) == (445, 281, 598, 338)
    assert x1 < x2 and y1 < y2


def test_load_annotations_raises_on_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "nao_existe.json"
    with pytest.raises(DatasetError):
        load_annotations(missing_path)


def test_load_annotations_raises_on_invalid_root_type(tmp_path) -> None:
    bad_file = tmp_path / "invalido.json"
    bad_file.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(DatasetError):
        load_annotations(bad_file)


def test_load_annotations_skips_invalid_records_without_raising(tmp_path) -> None:
    bad_file = tmp_path / "parcial.json"
    bad_file.write_text(
        """
        [
            {"filename": "ok.jpg", "make": "fiat", "model": "uno", "color": "white",
             "type": "car", "infrared": "no", "rear_view": "yes",
             "corners": [{"x":0,"y":0},{"x":10,"y":0},{"x":10,"y":10},{"x":0,"y":10}],
             "plate": "ABC1234"},
            {"filename": "sem_placa.jpg", "make": "fiat", "model": "uno", "color": "white",
             "type": "car", "infrared": "no", "rear_view": "yes",
             "corners": [{"x":0,"y":0},{"x":10,"y":0},{"x":10,"y":10},{"x":0,"y":10}],
             "plate": ""}
        ]
        """,
        encoding="utf-8",
    )
    annotations, report = load_annotations(bad_file)
    assert report.total == 2
    assert report.valid == 1
    assert report.invalid == 1
    assert len(annotations) == 1
    assert annotations[0].filename == "ok.jpg"


def test_build_manifest_returns_expected_columns_and_row_count(loaded_annotations) -> None:
    annotations, _ = loaded_annotations
    manifest = build_manifest(annotations)
    assert len(manifest) == 19
    expected_cols = {
        "filename", "plate", "make", "model", "color", "type",
        "infrared", "rear_view",
        "plate_bbox_x1", "plate_bbox_y1", "plate_bbox_x2", "plate_bbox_y2",
    }
    assert expected_cols.issubset(set(manifest.columns))


def test_validate_images_exist_reports_missing_when_dir_absent(loaded_annotations, tmp_path) -> None:
    annotations, _ = loaded_annotations
    manifest = build_manifest(annotations)
    result = validate_images_exist(manifest, tmp_path / "nao_existe")
    assert result["checked"] == 19
    assert result["found"] == 0
    assert result["missing"] == 19


def test_split_by_vehicle_has_no_leakage_between_splits(loaded_annotations) -> None:
    annotations, _ = loaded_annotations
    manifest = build_manifest(annotations)
    result = split_by_vehicle(manifest, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert set(result["split"].unique()).issubset({"train", "val", "test"})

    # Nenhuma placa pode aparecer em mais de um split
    plate_to_splits = result.groupby("plate")["split"].nunique()
    assert (plate_to_splits == 1).all(), "Vazamento detectado: placa presente em múltiplos splits"


def test_split_by_vehicle_is_deterministic_given_same_seed(loaded_annotations) -> None:
    annotations, _ = loaded_annotations
    manifest = build_manifest(annotations)
    result_a = split_by_vehicle(manifest, seed=123)
    result_b = split_by_vehicle(manifest, seed=123)
    assert result_a["split"].tolist() == result_b["split"].tolist()


def test_split_by_vehicle_raises_when_ratios_do_not_sum_to_one(loaded_annotations) -> None:
    annotations, _ = loaded_annotations
    manifest = build_manifest(annotations)
    with pytest.raises(DatasetError):
        split_by_vehicle(manifest, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
