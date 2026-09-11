"""Testes da Etapa 6 — codificação de rótulos (LabelEncoders)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.classification.label_encoder import LabelEncoders
from src.utils.exceptions import DatasetError


def _make_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"type": "car", "make": "fiat", "model": "uno", "color": "white"},
            {"type": "car", "make": "chevrolet", "model": "onix", "color": "black"},
            {"type": "truck", "make": "fiat", "model": "uno", "color": "unknown"},
        ]
    )


def test_fit_builds_sorted_deterministic_mappings() -> None:
    manifest = _make_manifest()
    encoders = LabelEncoders.fit(manifest)

    assert encoders.num_classes["type"] == 2  # car, truck
    assert encoders.num_classes["make"] == 2  # fiat, chevrolet
    assert encoders.num_classes["color"] == 3  # white, black, unknown

    # ordem alfabética determinística
    assert encoders.class_to_idx["type"] == {"car": 0, "truck": 1}


def test_encode_decode_roundtrip() -> None:
    manifest = _make_manifest()
    encoders = LabelEncoders.fit(manifest)

    idx = encoders.encode("make", "fiat")
    assert encoders.decode("make", idx) == "fiat"


def test_encode_raises_on_unknown_class() -> None:
    manifest = _make_manifest()
    encoders = LabelEncoders.fit(manifest)

    with pytest.raises(DatasetError):
        encoders.encode("make", "toyota")  # não estava no manifest de treino


def test_fit_raises_when_task_column_missing() -> None:
    manifest = _make_manifest().drop(columns=["color"])
    with pytest.raises(DatasetError):
        LabelEncoders.fit(manifest)


def test_save_and_load_roundtrip(tmp_path) -> None:
    manifest = _make_manifest()
    encoders = LabelEncoders.fit(manifest)

    path = tmp_path / "encoders.json"
    encoders.save(path)
    loaded = LabelEncoders.load(path)

    assert loaded.class_to_idx == encoders.class_to_idx
    assert loaded.num_classes == encoders.num_classes
    assert loaded.decode("type", loaded.encode("type", "car")) == "car"


def test_load_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(DatasetError):
        LabelEncoders.load(tmp_path / "nao_existe.json")


def test_fit_with_custom_task_subset() -> None:
    manifest = _make_manifest()
    encoders = LabelEncoders.fit(manifest, tasks=["type", "color"])
    assert set(encoders.class_to_idx.keys()) == {"type", "color"}
