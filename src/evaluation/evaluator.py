"""Rotinas de avaliação do OCR, do classificador e do pipeline completo,
rodando sobre o split de TESTE do manifest (nunca usado em treino nem em
seleção de modelo/early-stopping — ver docs/02_dataset.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2
import pandas as pd

from src.evaluation.metrics import (
    balanced_classification_accuracy,
    character_accuracy,
    classification_accuracy,
    exact_match_accuracy,
    known_format_rate,
)
from src.evaluation.schemas import AttributeMetrics, ClassificationMetrics, EndToEndMetrics, OCRMetrics
from src.utils.geometry import crop_with_margin
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CLASSIFICATION_TASKS = ("type", "make", "model", "color")


def evaluate_ocr_isolated(
    test_manifest: pd.DataFrame,
    images_dir: Union[str, Path],
    ocr_reader,
    margin_ratio: float = 0.05,
) -> OCRMetrics:
    """Avalia o OCR isoladamente, usando o recorte de placa via bbox
    GROUND-TRUTH (não via `PlateDetector`) — mede a capacidade "crua" do
    reconhecedor, isolando erros de detecção de placa.
    """
    images_dir = Path(images_dir)
    predictions: List[str] = []
    ground_truths: List[str] = []
    format_flags: List[bool] = []
    skipped = 0

    for row in test_manifest.itertuples(index=False):
        image_path = images_dir / row.filename
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        crop = crop_with_margin(
            image, int(row.plate_bbox_x1), int(row.plate_bbox_y1),
            int(row.plate_bbox_x2), int(row.plate_bbox_y2), margin_ratio,
        )
        result = ocr_reader.read(crop)

        predictions.append(result.normalized_text if result else "")
        ground_truths.append(row.plate)
        format_flags.append(result.matches_known_format if result else False)

    if skipped:
        logger.warning(f"{skipped} imagem(ns) não encontradas — puladas na avaliação de OCR")

    metrics = OCRMetrics(
        num_samples=len(predictions),
        exact_match_accuracy=exact_match_accuracy(predictions, ground_truths),
        character_accuracy=character_accuracy(predictions, ground_truths),
        known_format_rate=known_format_rate(format_flags),
    )
    logger.info(
        f"OCR isolado ({metrics.num_samples} amostras): "
        f"exact-match={metrics.exact_match_accuracy:.2%}, "
        f"char-accuracy={metrics.character_accuracy:.2%}"
    )
    return metrics


def evaluate_classifier_isolated(
    test_manifest: pd.DataFrame,
    images_dir: Union[str, Path],
    classifier,
) -> ClassificationMetrics:
    """Avalia o classificador multi-tarefa isoladamente, na IMAGEM COMPLETA
    (mesma distribuição de entrada usada no treino — Etapa 6).
    """
    images_dir = Path(images_dir)
    predictions = {task: [] for task in _CLASSIFICATION_TASKS}
    ground_truths = {task: [] for task in _CLASSIFICATION_TASKS}
    skipped = 0

    for row in test_manifest.itertuples(index=False):
        image_path = images_dir / row.filename
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        result = classifier.classify(image)
        predictions["type"].append(result.type.label)
        predictions["make"].append(result.make.label)
        predictions["model"].append(result.model.label)
        predictions["color"].append(result.color.label)
        for task in _CLASSIFICATION_TASKS:
            ground_truths[task].append(str(getattr(row, task)))

    if skipped:
        logger.warning(f"{skipped} imagem(ns) não encontradas — puladas na avaliação de classificação")

    per_attribute = {}
    for task in _CLASSIFICATION_TASKS:
        preds, gts = predictions[task], ground_truths[task]
        per_attribute[task] = AttributeMetrics(
            num_samples=len(preds),
            accuracy=classification_accuracy(preds, gts),
            balanced_accuracy=balanced_classification_accuracy(preds, gts),
            num_classes=len(set(gts)),
        )

    metrics = ClassificationMetrics(per_attribute=per_attribute)
    logger.info(
        f"Classificação isolada: "
        + ", ".join(f"{t}={m.accuracy:.2%}" for t, m in per_attribute.items())
        + f" | média={metrics.mean_accuracy:.2%}"
    )
    return metrics


def evaluate_pipeline_end_to_end(
    test_manifest: pd.DataFrame,
    images_dir: Union[str, Path],
    pipeline,
) -> EndToEndMetrics:
    """Avalia o sistema completo (Etapa 7), rodando a imagem bruta pelo
    pipeline ponta a ponta — inclui erros compostos de detecção de
    veículo/placa, diferente da avaliação isolada.

    Assume 1 veículo principal por imagem (premissa do próprio dataset —
    ver docs/02_dataset.md): quando o `VehicleDetector` encontra mais de
    um veículo, usa o de maior área de bbox como o veículo "principal" a
    ser comparado com o ground-truth da linha do manifest.
    """
    images_dir = Path(images_dir)
    plate_predictions: List[str] = []
    plate_ground_truths: List[str] = []
    classification_predictions = {task: [] for task in _CLASSIFICATION_TASKS}
    classification_ground_truths = {task: [] for task in _CLASSIFICATION_TASKS}
    detected_flags: List[bool] = []
    latencies_ms: List[float] = []
    skipped = 0

    for row in test_manifest.itertuples(index=False):
        image_path = images_dir / row.filename
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        frame_result = pipeline.process_frame(image)
        latencies_ms.append(frame_result.total_latency_ms)

        if not frame_result.vehicles:
            detected_flags.append(False)
            plate_predictions.append("")  # ausência de detecção conta como erro
            for task in _CLASSIFICATION_TASKS:
                classification_predictions[task].append("")
        else:
            detected_flags.append(True)
            primary = max(frame_result.vehicles, key=lambda v: v.vehicle_detection.area)
            plate_predictions.append(primary.plate_text or "")
            classification_predictions["type"].append(primary.classification.type.label)
            classification_predictions["make"].append(primary.classification.make.label)
            classification_predictions["model"].append(primary.classification.model.label)
            classification_predictions["color"].append(primary.classification.color.label)

        plate_ground_truths.append(row.plate)
        for task in _CLASSIFICATION_TASKS:
            classification_ground_truths[task].append(str(getattr(row, task)))

    if skipped:
        logger.warning(f"{skipped} imagem(ns) não encontradas — puladas na avaliação end-to-end")

    per_task_acc = [
        classification_accuracy(classification_predictions[task], classification_ground_truths[task])
        for task in _CLASSIFICATION_TASKS
    ]
    mean_classification_acc = sum(per_task_acc) / len(per_task_acc) if per_task_acc else 0.0
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    metrics = EndToEndMetrics(
        num_samples=len(plate_ground_truths),
        vehicle_detection_rate=(sum(detected_flags) / len(detected_flags) if detected_flags else 0.0),
        plate_exact_match_accuracy=exact_match_accuracy(plate_predictions, plate_ground_truths),
        classification_mean_accuracy=mean_classification_acc,
        avg_latency_ms=avg_latency,
        avg_fps=(1000.0 / avg_latency if avg_latency > 0 else 0.0),
    )
    logger.info(
        f"End-to-end ({metrics.num_samples} amostras): "
        f"detecção={metrics.vehicle_detection_rate:.2%}, "
        f"placa exact-match={metrics.plate_exact_match_accuracy:.2%}, "
        f"classificação média={metrics.classification_mean_accuracy:.2%}, "
        f"latência média={metrics.avg_latency_ms:.1f}ms"
    )
    return metrics
