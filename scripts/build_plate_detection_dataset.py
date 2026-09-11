"""Gera o dataset YOLO para treino do detector de placa, a partir do
manifest gerado na Etapa 2.

Uso:
    python scripts/build_plate_detection_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.preprocessing.yolo_converter import convert_manifest_to_yolo
from src.utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_PATH = Path("datasets/processed/manifest.csv")
OUTPUT_DIR = Path("datasets/processed/plate_detection")


def _load_base_config() -> dict:
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} não encontrado. Rode primeiro: "
            "python scripts/build_dataset_manifest.py"
        )

    config = _load_base_config()
    manifest = pd.read_csv(MANIFEST_PATH)
    images_dir = config["dataset"]["images_dir"]

    logger.info(f"Convertendo manifest ({len(manifest)} linhas) para formato YOLO em {OUTPUT_DIR}")
    report = convert_manifest_to_yolo(manifest, images_dir=images_dir, output_dir=OUTPUT_DIR)

    print(f"\nConvertidos: {report.converted}/{report.total} ({report.success_rate:.2%})")
    print(f"Imagens ausentes: {report.skipped_missing_image}")
    print(f"Imagens ilegíveis: {report.skipped_unreadable_image}")
    if report.skipped_details:
        print("\nPrimeiros itens pulados:")
        for detail in report.skipped_details[:10]:
            print(f"  - {detail}")


if __name__ == "__main__":
    main()
