"""Testes da Etapa 8 — funções puras de métrica."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    balanced_classification_accuracy,
    character_accuracy,
    classification_accuracy,
    exact_match_accuracy,
    known_format_rate,
)


def test_exact_match_accuracy_all_correct() -> None:
    assert exact_match_accuracy(["ABC1234", "DEF5678"], ["ABC1234", "DEF5678"]) == 1.0


def test_exact_match_accuracy_partial() -> None:
    assert exact_match_accuracy(["ABC1234", "WRONG"], ["ABC1234", "DEF5678"]) == 0.5


def test_exact_match_accuracy_empty_lists() -> None:
    assert exact_match_accuracy([], []) == 0.0


def test_exact_match_accuracy_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        exact_match_accuracy(["A"], ["A", "B"])


def test_character_accuracy_perfect_match() -> None:
    assert character_accuracy(["ABC1234"], ["ABC1234"]) == 1.0


def test_character_accuracy_one_character_wrong() -> None:
    # "ABC1234" vs "ABC1235" -> distância de edição = 1, 7 caracteres no gt
    acc = character_accuracy(["ABC1234"], ["ABC1235"])
    assert acc == pytest.approx(1 - 1 / 7)


def test_character_accuracy_is_more_lenient_than_exact_match() -> None:
    predictions = ["ABC1235"]  # 1 caractere errado
    ground_truths = ["ABC1234"]
    assert character_accuracy(predictions, ground_truths) > exact_match_accuracy(predictions, ground_truths)


def test_character_accuracy_completely_wrong_clips_at_zero() -> None:
    acc = character_accuracy(["ZZZZZZZZZZZZZZ"], ["A"])
    assert acc == 0.0


def test_character_accuracy_empty_lists() -> None:
    assert character_accuracy([], []) == 0.0


def test_classification_accuracy_matches_exact_match_logic() -> None:
    assert classification_accuracy(["car", "truck"], ["car", "bus"]) == 0.5


def test_balanced_classification_accuracy_penalizes_minority_class_errors() -> None:
    # 90 "car" corretos, 10 "truck" todos errados -> acc simples=90%, balanceada=45%
    predictions = ["car"] * 90 + ["car"] * 10
    ground_truths = ["car"] * 90 + ["truck"] * 10

    simple_acc = classification_accuracy(predictions, ground_truths)
    balanced_acc = balanced_classification_accuracy(predictions, ground_truths)

    assert simple_acc == pytest.approx(0.9)
    assert balanced_acc == pytest.approx(0.5)  # média(100%, 0%)
    assert balanced_acc < simple_acc


def test_balanced_classification_accuracy_equal_classes_matches_simple() -> None:
    predictions = ["car", "truck", "car", "truck"]
    ground_truths = ["car", "truck", "car", "truck"]
    assert balanced_classification_accuracy(predictions, ground_truths) == 1.0


def test_known_format_rate() -> None:
    assert known_format_rate([True, True, False, True]) == 0.75


def test_known_format_rate_empty() -> None:
    assert known_format_rate([]) == 0.0
