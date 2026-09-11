"""Avalia o detector de placa (Etapa 4) no split de TESTE, usando o
avaliador nativo do Ultralytics (mAP50, mAP50-95, precisão, recall) — não
reimplementamos cálculo de mAP, já que o `ultralytics.YOLO.val()` já faz
isso corretamente.

Requer:
    - models/checkpoints/plate_detector_best.pt (Etapa 4 treinada)
    - datasets/processed/plate_detection/data.yaml (Etapa 4)

Uso:
    python scripts/evaluate_plate_detector.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import get_logger

logger = get_logger(__name__)

WEIGHTS_PATH = Path("models/checkpoints/plate_detector_best.pt")
DATA_YAML = Path("datasets/processed/plate_detection/data.yaml")
OUTPUT_PATH = Path("outputs/reports/plate_detector_eval.json")


def main() -> None:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"{WEIGHTS_PATH} não encontrado. Rode primeiro: python scripts/train_plate_detector.py"
        )
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} não encontrado. Rode primeiro: python scripts/build_plate_detection_dataset.py"
        )

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS_PATH))
    logger.info(f"Avaliando {WEIGHTS_PATH} no split de teste de {DATA_YAML}")
    results = model.val(data=str(DATA_YAML), split="test")

    report = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nMétricas do detector de placa (split de teste):")
    for key, value in report.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nRelatório salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
