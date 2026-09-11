"""Funções de métrica para avaliação do OCR e da classificação de veículo.

Todas as funções aqui são puras (sem I/O, sem dependência de modelo),
o que as torna 100% testáveis com exemplos conhecidos.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def _levenshtein_distance(a: str, b: str) -> int:
    """Distância de edição (inserção/remoção/substituição) entre duas strings.

    Implementação O(len(a) * len(b)) em espaço O(min(len(a), len(b))) —
    suficiente para strings curtas como placas veiculares (6-7 caracteres).
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (char_a != char_b)
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
        previous_row = current_row

    return previous_row[-1]


def exact_match_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Fração de predições que batem EXATAMENTE com o ground-truth.

    Métrica oficial de OCR Accuracy do projeto (decisão registrada em
    docs/08_evaluation.md) — reflete o uso real: uma placa com um único
    caractere errado é tão inútil quanto uma completamente errada para
    fins de busca/identificação.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions e ground_truths devem ter o mesmo tamanho")
    if not predictions:
        return 0.0

    matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return matches / len(predictions)


def character_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Acurácia por caractere, baseada em distância de edição (1 - CER).

    Métrica complementar/diagnóstica (não-oficial) — mais permissiva que
    `exact_match_accuracy`, comum em benchmarks acadêmicos de OCR.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions e ground_truths devem ter o mesmo tamanho")
    if not predictions:
        return 0.0

    total_chars = sum(len(gt) for gt in ground_truths)
    if total_chars == 0:
        return 0.0

    total_errors = sum(
        _levenshtein_distance(pred, gt) for pred, gt in zip(predictions, ground_truths)
    )
    accuracy = 1.0 - (total_errors / total_chars)
    return max(0.0, accuracy)  # CER pode superar 100% em casos patológicos; limitamos em 0


def classification_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Acurácia simples (% de acertos) — métrica oficial de classificação
    do projeto (decisão registrada em docs/08_evaluation.md).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions e ground_truths devem ter o mesmo tamanho")
    if not predictions:
        return 0.0

    matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return matches / len(predictions)


def balanced_classification_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Acurácia balanceada (macro) — média da acurácia (recall) de cada
    classe individualmente, dando peso igual a classes raras e frequentes.

    Métrica complementar/diagnóstica (não-oficial) — usada para não
    mascarar mau desempenho em classes minoritárias (ver docs/06_classification.md,
    achado sobre desbalanceamento de `type` e `model`).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions e ground_truths devem ter o mesmo tamanho")
    if not predictions:
        return 0.0

    per_class_correct: Dict[str, int] = {}
    per_class_total: Dict[str, int] = {}
    for pred, gt in zip(predictions, ground_truths):
        per_class_total[gt] = per_class_total.get(gt, 0) + 1
        if pred == gt:
            per_class_correct[gt] = per_class_correct.get(gt, 0) + 1

    per_class_accuracy: List[float] = [
        per_class_correct.get(cls, 0) / total for cls, total in per_class_total.items()
    ]
    return sum(per_class_accuracy) / len(per_class_accuracy)


def known_format_rate(flags: Sequence[bool]) -> float:
    """Fração de predições de OCR cujo texto normalizado bate com um
    padrão de placa conhecido (ver `src.alpr.plate_format`).
    """
    if not flags:
        return 0.0
    return sum(1 for f in flags if f) / len(flags)
