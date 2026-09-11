"""Testes da Etapa 5 — leitura de texto (OCR) da placa.

Usa mocks para a saída do `paddleocr.TextRecognition`, seguindo o mesmo
padrão de `tests/test_vehicle_detector.py` e `tests/test_plate_detector.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.alpr.ocr_reader import OCRReader
from src.alpr.schemas import OCRResult
from src.utils.exceptions import ModelLoadError, OCRError


class _FakeTextRecognitionModel:
    def __init__(self, rec_text: str, rec_score: float) -> None:
        self._rec_text = rec_text
        self._rec_score = rec_score

    def predict(self, input, **kwargs):
        return [{"rec_text": self._rec_text, "rec_score": self._rec_score}]


def _make_reader(fake_model) -> OCRReader:
    return OCRReader(device="cpu", _model=fake_model)


def test_read_returns_ocr_result_with_normalized_text() -> None:
    fake_model = _FakeTextRecognitionModel(rec_text="abc-1234", rec_score=0.95)
    reader = _make_reader(fake_model)

    result = reader.read(np.zeros((40, 120, 3), dtype=np.uint8))

    assert isinstance(result, OCRResult)
    assert result.text == "abc-1234"
    assert result.normalized_text == "ABC1234"
    assert result.confidence == pytest.approx(0.95)
    assert result.matches_known_format is True


def test_read_flags_unrecognized_format() -> None:
    fake_model = _FakeTextRecognitionModel(rec_text="??##", rec_score=0.4)
    reader = _make_reader(fake_model)

    result = reader.read(np.zeros((40, 120, 3), dtype=np.uint8))

    assert result.matches_known_format is False


def test_read_returns_none_for_empty_image() -> None:
    fake_model = _FakeTextRecognitionModel(rec_text="ABC1234", rec_score=0.9)
    reader = _make_reader(fake_model)

    assert reader.read(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert reader.read(None) is None


def test_read_raises_ocr_error_on_model_failure() -> None:
    class _BrokenModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("falha simulada")

    reader = OCRReader(device="cpu", _model=_BrokenModel())
    with pytest.raises(OCRError):
        reader.read(np.zeros((40, 120, 3), dtype=np.uint8))


def test_load_model_raises_when_model_dir_does_not_exist(tmp_path) -> None:
    missing_dir = tmp_path / "nao_existe"
    with pytest.raises(ModelLoadError):
        OCRReader(model_dir=str(missing_dir), device="cpu")


def test_from_config_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(ModelLoadError):
        OCRReader.from_config(tmp_path / "nao_existe.yaml")


def test_from_config_loads_expected_parameters(monkeypatch) -> None:
    """Valida leitura de configs/plate_recognition.yaml sem carregar pesos reais."""
    fake_model = _FakeTextRecognitionModel(rec_text="X", rec_score=0.5)
    monkeypatch.setattr(
        OCRReader, "_load_model", staticmethod(lambda model_dir, model_name, device: fake_model)
    )

    reader = OCRReader.from_config("configs/plate_recognition.yaml")

    assert reader.model_name == "PP-OCRv5_mobile_rec"
    assert reader.model_dir == "models/checkpoints/plate_recognizer"
    assert reader.device == "cpu"
