"""Testes da Etapa 5 — validação de formato de placa (BR/Mercosul)."""

from __future__ import annotations

import pytest

from src.alpr.plate_format import matches_known_plate_format, normalize_plate_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("abc1234", "ABC1234"),
        ("ABC-1234", "ABC1234"),
        (" abc 1234 ", "ABC1234"),
        ("aBc1d23", "ABC1D23"),
    ],
)
def test_normalize_plate_text(raw: str, expected: str) -> None:
    assert normalize_plate_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    ["ABC1234", "abc1234", "ABC-1234", "ABC1D23", "abc1d23"],
)
def test_matches_known_plate_format_valid_cases(text: str) -> None:
    assert matches_known_plate_format(text) is True


@pytest.mark.parametrize(
    "text",
    ["", "AB1234", "ABCD1234", "ABC12345", "1234ABC", "ABC12D3"],
)
def test_matches_known_plate_format_invalid_cases(text: str) -> None:
    assert matches_known_plate_format(text) is False
