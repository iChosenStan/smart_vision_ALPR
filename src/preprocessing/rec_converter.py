"""Conversão do manifest (Etapa 2) para o formato de dataset de
reconhecimento de texto (OCR) esperado pelo PaddleX/PaddleOCR.

Gera a estrutura:

    output_dir/
    ├── images/{train,val,test}/*.jpg   (recorte APERTADO da placa)
    ├── train.txt                        ("images/train/xxx.jpg\tTEXTO")
    ├── val.txt
    └── test.txt

Diferente do `yolo_converter` (Etapa 4), aqui a imagem salva já é o
**recorte da placa** (não a imagem inteira) — o modelo de reconhecimento
espera receber apenas a região de texto, sem precisar localizá-la.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import cv2
import pandas as pd

from src.utils.exceptions import DatasetError
from src.utils.geometry import crop_with_margin
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SPLITS = ("train", "val", "test")


@dataclass
class RecConversionReport:
    """Resumo da conversão manifest → dataset de reconhecimento."""

    total: int
    converted: int
    skipped_missing_image: int
    skipped_unreadable_image: int
    skipped_empty_crop: int
    skipped_details: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.converted / self.total if self.total else 0.0


def _write_char_dict(output_dir: Path, plates: pd.Series) -> Path:
    """Gera `dict.txt` — dicionário de caracteres exigido pelo PaddleOCR/PaddleX
    para treino de reconhecimento (um caractere por linha).

    Descoberto na prática (Etapa 5, execução real no Colab/local): o
    PaddleX espera esse arquivo em `{dataset_dir}/dict.txt` por padrão, e
    o treino falha com `FileNotFoundError` sem ele. Construído a partir
    dos caracteres realmente usados nas placas do manifest, para cobrir
    qualquer variação (não assume apenas A-Z0-9, embora seja o esperado
    para placas BR/Mercosul).
    """
    chars = sorted({ch for plate in plates for ch in str(plate)})
    dict_path = output_dir / "dict.txt"
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chars) + "\n")
    logger.info(f"dict.txt gerado com {len(chars)} caracteres em {dict_path}")
    return dict_path


def convert_manifest_to_rec_dataset(
    manifest: pd.DataFrame,
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    margin_ratio: float = 0.05,
) -> RecConversionReport:
    """Converte o manifest em um dataset de reconhecimento de texto.

    Args:
        manifest: DataFrame com colunas `filename`, `plate`,
            `plate_bbox_x1..y2` e `split`.
        images_dir: Diretório com as imagens originais (não-recortadas).
        output_dir: Diretório onde o dataset de reconhecimento será gerado.
        margin_ratio: Margem extra ao redor do bbox da placa (fração da
            largura/altura), para não cortar bordas de caracteres.

    Returns:
        Relatório de conversão.

    Raises:
        DatasetError: Se o manifest não tiver a coluna `split`.
    """
    if "split" not in manifest.columns:
        raise DatasetError("Manifest não possui coluna 'split' — rode split_by_vehicle antes.")

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    for split in _SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    label_lines = {split: [] for split in _SPLITS}
    converted = 0
    skipped_missing = 0
    skipped_unreadable = 0
    skipped_empty = 0
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

        crop = crop_with_margin(
            image,
            int(row.plate_bbox_x1),
            int(row.plate_bbox_y1),
            int(row.plate_bbox_x2),
            int(row.plate_bbox_y2),
            margin_ratio,
        )
        if crop.size == 0:
            skipped_empty += 1
            skipped_details.append(f"[{row.filename}] recorte de placa vazio (bbox inválido)")
            continue

        dst_image = output_dir / "images" / row.split / row.filename
        cv2.imwrite(str(dst_image), crop)

        rel_path = f"images/{row.split}/{row.filename}"
        label_lines[row.split].append(f"{rel_path}\t{row.plate}")
        converted += 1

    for split in _SPLITS:
        label_path = output_dir / f"{split}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines[split]) + ("\n" if label_lines[split] else ""))

    _write_char_dict(output_dir, manifest["plate"])

    report = RecConversionReport(
        total=len(manifest),
        converted=converted,
        skipped_missing_image=skipped_missing,
        skipped_unreadable_image=skipped_unreadable,
        skipped_empty_crop=skipped_empty,
        skipped_details=skipped_details,
    )

    logger.info(
        f"Conversão do dataset de reconhecimento concluída: "
        f"{report.converted}/{report.total} convertidos "
        f"({report.skipped_missing_image} imagem ausente, "
        f"{report.skipped_unreadable_image} ilegível, "
        f"{report.skipped_empty_crop} recorte vazio)"
    )
    if skipped_details:
        logger.warning(f"Primeiros itens pulados: {skipped_details[:5]}")

    return report
