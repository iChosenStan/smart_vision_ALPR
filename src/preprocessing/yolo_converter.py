"""Conversão do manifest (Etapa 2) para o formato de dataset YOLO,
necessário para treinar o detector de placa (Etapa 4).

Gera a estrutura padrão esperada por `ultralytics.YOLO(...).train()`:

    output_dir/
    ├── images/{train,val,test}/*.jpg
    ├── labels/{train,val,test}/*.txt   (formato: "0 xc yc w h", normalizado)
    └── data.yaml

O bbox usado é o axis-aligned (`plate_bbox_x1..y2`, calculado na Etapa 2 a
partir do quadrilátero original da placa) — uma simplificação padrão para
detectores baseados em retângulo (ver docs/04_plate_detection.md).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import pandas as pd
import yaml

from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CLASS_ID = 0
_CLASS_NAME = "plate"
_SPLITS = ("train", "val", "test")


@dataclass
class ConversionReport:
    """Resumo da conversão manifest → formato YOLO."""

    total: int
    converted: int
    skipped_missing_image: int
    skipped_unreadable_image: int
    skipped_details: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.converted / self.total if self.total else 0.0


def _to_yolo_bbox(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Converte bbox absoluto (x1,y1,x2,y2) para formato YOLO normalizado
    (x_center, y_center, width, height), todos em [0, 1].
    """
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def convert_manifest_to_yolo(
    manifest: pd.DataFrame,
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    copy_mode: str = "copy",
) -> ConversionReport:
    """Converte o manifest em uma estrutura de dataset YOLO por split.

    Args:
        manifest: DataFrame com colunas `filename`, `plate_bbox_x1..y2` e
            `split` (gerado por `dataset_registry.split_by_vehicle`).
        images_dir: Diretório contendo as imagens originais (.jpg).
        output_dir: Diretório onde a estrutura YOLO será gerada.
        copy_mode: "copy" copia os arquivos de imagem; "symlink" cria links
            simbólicos (mais rápido, economiza espaço em disco — útil no Colab).

    Returns:
        Relatório de conversão.

    Raises:
        DatasetError: Se o manifest não tiver a coluna `split`, ou se
            `copy_mode` for inválido.
    """
    if "split" not in manifest.columns:
        raise DatasetError("Manifest não possui coluna 'split' — rode split_by_vehicle antes.")
    if copy_mode not in ("copy", "symlink"):
        raise DatasetError(f"copy_mode inválido: {copy_mode!r} (use 'copy' ou 'symlink')")

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    for split in _SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_missing = 0
    skipped_unreadable = 0
    skipped_details: List[str] = []

    for row in manifest.itertuples(index=False):
        src_image = images_dir / row.filename
        if not src_image.exists():
            skipped_missing += 1
            skipped_details.append(f"[{row.filename}] imagem não encontrada em {images_dir}")
            continue

        image = cv2.imread(str(src_image))
        if image is None:
            skipped_unreadable += 1
            skipped_details.append(f"[{row.filename}] falha ao ler imagem (arquivo corrompido?)")
            continue

        img_h, img_w = image.shape[:2]
        xc, yc, w, h = _to_yolo_bbox(
            row.plate_bbox_x1, row.plate_bbox_y1, row.plate_bbox_x2, row.plate_bbox_y2, img_w, img_h
        )

        dst_image = output_dir / "images" / row.split / row.filename
        dst_label = output_dir / "labels" / row.split / (Path(row.filename).stem + ".txt")

        if copy_mode == "symlink":
            if dst_image.exists() or dst_image.is_symlink():
                dst_image.unlink()
            dst_image.symlink_to(src_image.resolve())
        else:
            shutil.copy2(src_image, dst_image)

        with open(dst_label, "w", encoding="utf-8") as f:
            f.write(f"{_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        converted += 1

    report = ConversionReport(
        total=len(manifest),
        converted=converted,
        skipped_missing_image=skipped_missing,
        skipped_unreadable_image=skipped_unreadable,
        skipped_details=skipped_details,
    )

    logger.info(
        f"Conversão YOLO concluída: {report.converted}/{report.total} convertidos "
        f"({report.skipped_missing_image} imagem ausente, "
        f"{report.skipped_unreadable_image} ilegível)"
    )
    if skipped_details:
        logger.warning(f"Primeiros itens pulados: {skipped_details[:5]}")

    write_data_yaml(output_dir)

    return report


def write_data_yaml(output_dir: Union[str, Path]) -> Path:
    """Gera o `data.yaml` esperado pelo `ultralytics.YOLO(...).train()`."""
    output_dir = Path(output_dir)
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": [_CLASS_NAME],
    }
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    logger.info(f"data.yaml gerado em {data_yaml_path}")
    return data_yaml_path
