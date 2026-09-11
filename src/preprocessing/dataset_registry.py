"""Registro/organização do dataset: manifest tabular, validação de
integridade (imagens existem?) e split train/val/test.

O split é agrupado por `plate` (identificador único do veículo), evitando
que o mesmo veículo apareça em mais de um split — o UFPR-VeSV possui
16.297 veículos únicos capturados em 24.945 imagens (múltiplas capturas
do mesmo veículo em diferentes momentos/ângulos).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from src.preprocessing.schemas import PlateAnnotation
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_SPLITS = {"train", "val", "test"}


def build_manifest(annotations: List[PlateAnnotation]) -> pd.DataFrame:
    """Converte a lista de anotações em um DataFrame tabular.

    O DataFrame resultante é a base para EDA, geração de splits e,
    futuramente, conversão para formatos de treino (ex: YOLO).
    """
    rows = []
    for ann in annotations:
        x1, y1, x2, y2 = ann.bbox_xyxy
        rows.append(
            {
                "filename": ann.filename,
                "plate": ann.plate,
                "make": ann.make,
                "model": ann.model,
                "color": ann.color,
                "type": ann.type,
                "infrared": ann.infrared,
                "rear_view": ann.rear_view,
                "plate_bbox_x1": x1,
                "plate_bbox_y1": y1,
                "plate_bbox_x2": x2,
                "plate_bbox_y2": y2,
            }
        )
    manifest = pd.DataFrame(rows)
    logger.info(
        f"Manifest construído: {len(manifest)} imagens, "
        f"{manifest['plate'].nunique()} veículos únicos"
    )
    return manifest


def validate_images_exist(manifest: pd.DataFrame, images_dir: Union[str, Path]) -> Dict[str, int]:
    """Verifica quantas imagens referenciadas no manifest existem em disco.

    Não lança exceção em caso de imagens ausentes (dataset pode estar
    parcialmente baixado durante desenvolvimento) — apenas reporta.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        logger.warning(
            f"Diretório de imagens não encontrado: {images_dir}. "
            "Pulando validação de existência."
        )
        return {"checked": len(manifest), "found": 0, "missing": len(manifest)}

    found = sum(1 for fname in manifest["filename"] if (images_dir / fname).exists())
    missing = len(manifest) - found

    if missing:
        logger.warning(
            f"{missing}/{len(manifest)} imagens referenciadas nas anotações "
            f"não foram encontradas em {images_dir}"
        )
    else:
        logger.info(f"Todas as {found} imagens do manifest foram encontradas em {images_dir}")

    return {"checked": len(manifest), "found": found, "missing": missing}


def split_by_vehicle(
    manifest: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Divide o manifest em train/val/test agrupando por veículo (placa).

    Garante que todas as imagens do mesmo veículo caiam no mesmo split,
    evitando vazamento de dados (data leakage) entre treino e avaliação.

    Args:
        manifest: DataFrame retornado por `build_manifest`.
        train_ratio: Proporção de veículos para treino.
        val_ratio: Proporção de veículos para validação.
        test_ratio: Proporção de veículos para teste.
        seed: Semente para embaralhamento determinístico.

    Returns:
        Cópia do manifest com uma coluna adicional `split`
        (valores: "train", "val" ou "test").

    Raises:
        DatasetError: Se as proporções não somarem 1.0.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise DatasetError(
            f"As proporções de split devem somar 1.0 (recebido: {total_ratio})"
        )

    unique_plates = manifest["plate"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(unique_plates)

    n = len(unique_plates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_plates = set(unique_plates[:n_train])
    val_plates = set(unique_plates[n_train : n_train + n_val])
    # o restante vai para teste, evitando perda de veículos por arredondamento
    test_plates = set(unique_plates[n_train + n_val :])

    def assign(plate: str) -> str:
        if plate in train_plates:
            return "train"
        if plate in val_plates:
            return "val"
        return "test"

    result = manifest.copy()
    result["split"] = result["plate"].apply(assign)

    image_counts = result["split"].value_counts().to_dict()
    logger.info(
        "Split por veículo (sem vazamento) — "
        f"imagens: {image_counts} | "
        f"veículos: train={len(train_plates)}, val={len(val_plates)}, test={len(test_plates)}"
    )

    return result
