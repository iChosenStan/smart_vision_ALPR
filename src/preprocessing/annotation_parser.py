"""Parser do arquivo de anotações do UFPR-VeSV (`annotations.json`).

Formato de entrada esperado (lista de registros no JSON raiz):

    [
        {
            "filename": "img_00001.jpg",
            "make": "chevrolet",
            "model": "pickup_corsa",
            "color": "unknown",
            "type": "compact-pickup",
            "infrared": "yes",
            "rear_view": "yes",
            "corners": [{"x": 445, "y": 290}, ...],  # 4 pontos
            "plate": "MAV6C14"
        },
        ...
    ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from src.preprocessing.schemas import PlateAnnotation, Point
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_YES_NO_MAP = {"yes": True, "no": False}


@dataclass
class ParseReport:
    """Resumo do processo de parsing, incluindo registros inválidos."""

    total: int
    valid: int
    invalid: int
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.valid / self.total if self.total else 0.0


def _parse_bool(value: Any, field_name: str, filename: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in _YES_NO_MAP:
        raise DatasetError(
            f"[{filename}] valor inválido para '{field_name}': {value!r} "
            f"(esperado 'yes' ou 'no')"
        )
    return _YES_NO_MAP[normalized]


def _parse_corners(raw_corners: Any, filename: str) -> Tuple[Point, Point, Point, Point]:
    if not isinstance(raw_corners, list) or len(raw_corners) != 4:
        raise DatasetError(
            f"[{filename}] 'corners' deve conter exatamente 4 pontos, "
            f"recebido: {raw_corners!r}"
        )
    try:
        points = tuple(Point(x=int(c["x"]), y=int(c["y"])) for c in raw_corners)
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(f"[{filename}] corner com formato inválido: {exc}") from exc
    return points  # type: ignore[return-value]


def _parse_record(raw: Dict[str, Any]) -> PlateAnnotation:
    filename = raw.get("filename")
    if not filename:
        raise DatasetError("Registro sem 'filename'")

    plate = str(raw.get("plate", "")).strip().upper()
    if not plate:
        raise DatasetError(f"[{filename}] campo 'plate' vazio")

    return PlateAnnotation(
        filename=filename,
        make=str(raw.get("make", "unknown")).strip().lower(),
        model=str(raw.get("model", "unknown")).strip().lower(),
        color=str(raw.get("color", "unknown")).strip().lower(),
        type=str(raw.get("type", "unknown")).strip().lower(),
        infrared=_parse_bool(raw.get("infrared", "no"), "infrared", filename),
        rear_view=_parse_bool(raw.get("rear_view", "no"), "rear_view", filename),
        corners=_parse_corners(raw.get("corners"), filename),
        plate=plate,
    )


def load_annotations(path: Union[str, Path]) -> Tuple[List[PlateAnnotation], ParseReport]:
    """Carrega e valida o arquivo de anotações do UFPR-VeSV.

    Registros inválidos não interrompem o carregamento — são reportados em
    `ParseReport.errors` e excluídos da lista retornada, permitindo que o
    pipeline continue funcionando mesmo com pequenas inconsistências no
    dataset original.

    Args:
        path: Caminho para o arquivo `annotations.json`.

    Returns:
        Tupla (lista de anotações válidas, relatório de parsing).

    Raises:
        DatasetError: Se o arquivo não existir ou o JSON raiz não for uma lista.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Arquivo de anotações não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise DatasetError(
            "Formato inesperado: esperava uma lista de registros no JSON raiz, "
            f"recebido: {type(raw_data).__name__}"
        )

    annotations: List[PlateAnnotation] = []
    errors: List[str] = []

    for raw in raw_data:
        try:
            annotations.append(_parse_record(raw))
        except DatasetError as exc:
            errors.append(str(exc))
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"[{raw.get('filename', '?')}] erro inesperado: {exc}")

    report = ParseReport(
        total=len(raw_data),
        valid=len(annotations),
        invalid=len(errors),
        errors=errors,
    )

    logger.info(
        f"Anotações carregadas: {report.valid}/{report.total} válidas "
        f"({report.success_rate:.2%})"
    )
    if errors:
        logger.warning(f"{len(errors)} registro(s) inválido(s). Primeiros 5: {errors[:5]}")

    return annotations, report
