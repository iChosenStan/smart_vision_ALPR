"""Testes da Etapa 6 — VehicleClassifier (wrapper de inferência).

Usa uma instância REAL (pequena) de `VehicleMultiTaskNet` injetada via
`_model`, evitando o download de pesos fine-tunados — testa a mecânica
completa de pré-processamento + forward + decodificação de rótulos.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.classification.label_encoder import LabelEncoders
from src.classification.multi_task_model import VehicleMultiTaskNet
from src.classification.schemas import VehicleClassification
from src.classification.vehicle_classifier import VehicleClassifier
from src.utils.exceptions import ClassificationError, ModelLoadError

_CLASS_TO_IDX = {
    "type": {"car": 0, "truck": 1},
    "make": {"fiat": 0, "ford": 1},
    "model": {"uno": 0, "f250": 1},
    "color": {"white": 0, "red": 1},
}


def _make_encoders() -> LabelEncoders:
    idx_to_class = {
        task: {idx: label for label, idx in mapping.items()} for task, mapping in _CLASS_TO_IDX.items()
    }
    return LabelEncoders(class_to_idx=_CLASS_TO_IDX, idx_to_class=idx_to_class)


def _make_classifier() -> VehicleClassifier:
    encoders = _make_encoders()
    model = VehicleMultiTaskNet(num_classes=encoders.num_classes, pretrained=False)
    model.eval()
    return VehicleClassifier(device="cpu", _model=model, _encoders=encoders)


def test_classify_returns_vehicle_classification_with_all_attributes() -> None:
    classifier = _make_classifier()
    fake_image = np.zeros((300, 400, 3), dtype=np.uint8)

    result = classifier.classify(fake_image)

    assert isinstance(result, VehicleClassification)
    assert result.type.label in {"car", "truck"}
    assert result.make.label in {"fiat", "ford"}
    assert result.model.label in {"uno", "f250"}
    assert result.color.label in {"white", "red"}
    for attr in (result.type, result.make, result.model, result.color):
        assert 0.0 <= attr.confidence <= 1.0


def test_classify_is_deterministic_in_eval_mode() -> None:
    classifier = _make_classifier()
    fake_image = np.full((300, 400, 3), 128, dtype=np.uint8)

    result_a = classifier.classify(fake_image)
    result_b = classifier.classify(fake_image)

    assert result_a.type.label == result_b.type.label
    assert result_a.type.confidence == pytest.approx(result_b.type.confidence)


def test_classify_raises_classification_error_on_model_failure() -> None:
    class _BrokenModel:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("falha simulada")

    encoders = _make_encoders()
    classifier = VehicleClassifier(device="cpu", _model=_BrokenModel(), _encoders=encoders)

    with pytest.raises(ClassificationError):
        classifier.classify(np.zeros((100, 100, 3), dtype=np.uint8))


def test_load_model_raises_when_weights_file_does_not_exist(tmp_path) -> None:
    encoders = _make_encoders()
    missing_weights = tmp_path / "nao_existe.pt"
    with pytest.raises(ModelLoadError):
        VehicleClassifier(weights=str(missing_weights), device="cpu", _encoders=encoders)


def test_from_config_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(ModelLoadError):
        VehicleClassifier.from_config(tmp_path / "nao_existe.yaml")


def test_from_config_loads_expected_parameters(monkeypatch) -> None:
    encoders = _make_encoders()
    model = VehicleMultiTaskNet(num_classes=encoders.num_classes, pretrained=False)
    monkeypatch.setattr(VehicleClassifier, "_load_model", staticmethod(lambda w, e, d: model))
    monkeypatch.setattr(LabelEncoders, "load", classmethod(lambda cls, path: encoders))

    classifier = VehicleClassifier.from_config("configs/classification.yaml")

    assert classifier.weights == "models/checkpoints/vehicle_classifier_best.pt"
    assert classifier.image_size == 224
