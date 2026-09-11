"""Script de linha de comando: gera o manifest do dataset UFPR-VeSV.

Lê `annotations.json`, valida os registros, gera o manifest com split
train/val/test (agrupado por veículo) e salva em `datasets/processed/`.

Uso:
    python scripts/build_dataset_manifest.py \
        --annotations datasets/raw/annotations.json \
        --images-dir datasets/raw/images \
        --output datasets/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path, permitindo rodar este
# script diretamente (ex: `python scripts/build_dataset_manifest.py`) sem
# precisar configurar PYTHONPATH manualmente — importante para uso no Colab.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.preprocessing.annotation_parser import load_annotations
from src.preprocessing.dataset_registry import (
    build_manifest,
    split_by_vehicle,
    validate_images_exist,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_base_config() -> dict:
    config_path = Path("configs/base.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    config = _load_base_config()
    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["split"]

    parser = argparse.ArgumentParser(description="Gera o manifest do dataset UFPR-VeSV.")
    parser.add_argument("--annotations", default=dataset_cfg["annotations_file"])
    parser.add_argument("--images-dir", default=dataset_cfg["images_dir"])
    parser.add_argument("--output", default="datasets/processed/manifest.csv")
    parser.add_argument("--train-ratio", type=float, default=split_cfg["train_ratio"])
    parser.add_argument("--val-ratio", type=float, default=split_cfg["val_ratio"])
    parser.add_argument("--test-ratio", type=float, default=split_cfg["test_ratio"])
    parser.add_argument("--seed", type=int, default=config["project"]["seed"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Carregando anotações de {args.annotations}")
    annotations, report = load_annotations(args.annotations)
    logger.info(f"{report.valid}/{report.total} registros válidos ({report.success_rate:.2%})")

    manifest = build_manifest(annotations)
    validate_images_exist(manifest, args.images_dir)

    manifest = split_by_vehicle(
        manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    logger.info(f"Manifest salvo em {output_path} ({len(manifest)} linhas)")

    print("\nResumo do split:")
    print(manifest["split"].value_counts())


if __name__ == "__main__":
    main()
