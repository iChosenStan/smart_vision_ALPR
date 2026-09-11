"""Script mestre de avaliação (Etapa 8): roda avaliação isolada do OCR e
do classificador, além da avaliação ponta a ponta do pipeline completo,
no split de TESTE (nunca usado em treino/seleção de modelo).

Gera um relatório em Markdown + JSON comparando os resultados às metas
definidas em `configs/base.yaml` (`pipeline_targets`).

Requer:
    - datasets/processed/manifest.csv (Etapa 2)
    - Imagens reais em datasets/raw/images/ (Etapa 2)
    - Todos os modelos treinados: PlateDetector (4), OCRReader (5),
      VehicleClassifier (6)

Uso:
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.alpr.ocr_reader import OCRReader
from src.classification.vehicle_classifier import VehicleClassifier
from src.evaluation.evaluator import (
    evaluate_classifier_isolated,
    evaluate_ocr_isolated,
    evaluate_pipeline_end_to_end,
)
from src.pipeline.smart_vision_pipeline import SmartVisionPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_PATH = Path("datasets/processed/manifest.csv")
REPORT_DIR = Path("outputs/reports")


def _load_targets() -> dict:
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["pipeline_targets"]


def _render_markdown(report: dict, targets: dict) -> str:
    lines = [
        "# Relatório de Avaliação — SmartVision ALPR",
        "",
        f"Gerado em: {report['generated_at']}",
        "",
        "## OCR (isolado — recorte via bbox ground-truth)",
        "",
        f"- Amostras: {report['ocr']['num_samples']}",
        f"- **Exact-match accuracy (oficial): {report['ocr']['exact_match_accuracy']:.2%}** "
        f"(meta: ≥{targets['ocr_accuracy_min']:.0%}) "
        f"— {'✅ OK' if report['ocr']['exact_match_accuracy'] >= targets['ocr_accuracy_min'] else '❌ ABAIXO DA META'}",
        f"- Acurácia por caractere (diagnóstico): {report['ocr']['character_accuracy']:.2%}",
        f"- Taxa de formato conhecido (BR/Mercosul): {report['ocr']['known_format_rate']:.2%}",
        "",
        "## Classificação (isolada — imagem completa)",
        "",
    ]
    for task, m in report["classification"].items():
        lines.append(
            f"- `{task}` ({m['num_classes']} classes): "
            f"acurácia={m['accuracy']:.2%}, balanceada={m['balanced_accuracy']:.2%}"
        )
    mean_acc = report["classification_mean_accuracy"]
    lines += [
        "",
        f"- **Acurácia média (oficial): {mean_acc:.2%}** (meta: ≥{targets['classification_accuracy_min']:.0%}) "
        f"— {'✅ OK' if mean_acc >= targets['classification_accuracy_min'] else '❌ ABAIXO DA META'}",
        "",
        "## Ponta a Ponta (pipeline completo, Etapa 7)",
        "",
        f"- Amostras: {report['end_to_end']['num_samples']}",
        f"- Taxa de detecção de veículo: {report['end_to_end']['vehicle_detection_rate']:.2%}",
        f"- Placa exact-match (end-to-end): {report['end_to_end']['plate_exact_match_accuracy']:.2%}",
        f"- Classificação média (end-to-end): {report['end_to_end']['classification_mean_accuracy']:.2%}",
        f"- **Latência média: {report['end_to_end']['avg_latency_ms']:.1f}ms** "
        f"(meta: ≤{targets['inference_latency_ms_max']}ms) "
        f"— {'✅ OK' if report['end_to_end']['avg_latency_ms'] <= targets['inference_latency_ms_max'] else '❌ ABAIXO DA META'}",
        f"- FPS médio: {report['end_to_end']['avg_fps']:.1f} (meta: ≥{targets['target_fps']})",
        "",
        "> Nota: a avaliação end-to-end tende a ter números mais baixos que "
        "a isolada — ela inclui erros compostos de detecção de veículo/placa, "
        "além de um descompasso conhecido de distribuição (ver docs/07_pipeline.md).",
    ]
    return "\n".join(lines)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} não encontrado. Rode primeiro: python scripts/build_dataset_manifest.py"
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    logger.info(f"Avaliando no split de teste: {len(test_manifest)} imagens")

    images_dir = "datasets/raw/images"
    targets = _load_targets()

    ocr_reader = OCRReader.from_config()
    ocr_metrics = evaluate_ocr_isolated(test_manifest, images_dir, ocr_reader)

    classifier = VehicleClassifier.from_config()
    classification_metrics = evaluate_classifier_isolated(test_manifest, images_dir, classifier)

    pipeline = SmartVisionPipeline.from_config()
    end_to_end_metrics = evaluate_pipeline_end_to_end(test_manifest, images_dir, pipeline)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ocr": {
            "num_samples": ocr_metrics.num_samples,
            "exact_match_accuracy": ocr_metrics.exact_match_accuracy,
            "character_accuracy": ocr_metrics.character_accuracy,
            "known_format_rate": ocr_metrics.known_format_rate,
        },
        "classification": {
            task: {
                "num_samples": m.num_samples,
                "accuracy": m.accuracy,
                "balanced_accuracy": m.balanced_accuracy,
                "num_classes": m.num_classes,
            }
            for task, m in classification_metrics.per_attribute.items()
        },
        "classification_mean_accuracy": classification_metrics.mean_accuracy,
        "end_to_end": {
            "num_samples": end_to_end_metrics.num_samples,
            "vehicle_detection_rate": end_to_end_metrics.vehicle_detection_rate,
            "plate_exact_match_accuracy": end_to_end_metrics.plate_exact_match_accuracy,
            "classification_mean_accuracy": end_to_end_metrics.classification_mean_accuracy,
            "avg_latency_ms": end_to_end_metrics.avg_latency_ms,
            "avg_fps": end_to_end_metrics.avg_fps,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "evaluation_report.json"
    md_path = REPORT_DIR / "evaluation_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(report, targets))

    print(f"\nRelatórios salvos em:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
