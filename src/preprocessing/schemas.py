"""Modelos de dados (schemas) das anotações do UFPR-VeSV.

O dataset fornece, por imagem, os atributos do veículo (marca, modelo,
cor, tipo) e o quadrilátero (4 pontos) delimitando a placa — não há
bounding box do veículo completo nas anotações originais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Point:
    """Ponto 2D em coordenadas de pixel (origem no canto superior esquerdo)."""

    x: int
    y: int


@dataclass(frozen=True)
class PlateAnnotation:
    """Anotação completa de uma imagem do UFPR-VeSV.

    Attributes:
        filename: Nome do arquivo de imagem (ex: "img_00001.jpg").
        make: Marca do veículo, normalizada em minúsculas.
        model: Modelo do veículo, normalizado em minúsculas.
        color: Cor do veículo, normalizada em minúsculas (pode ser "unknown").
        type: Categoria do veículo (ex: "car", "SUV", "motorcycle").
        infrared: Se a imagem foi capturada em modo infravermelho.
        rear_view: Se a imagem é da traseira do veículo (vs. frontal).
        corners: Quadrilátero (4 pontos, sentido não garantido pelo dataset
            original) delimitando a placa na imagem.
        plate: Texto da placa. Funciona como identificador único do veículo
            no dataset (o mesmo veículo pode aparecer em múltiplas imagens).
    """

    filename: str
    make: str
    model: str
    color: str
    type: str
    infrared: bool
    rear_view: bool
    corners: Tuple[Point, Point, Point, Point]
    plate: str

    @property
    def bbox_xyxy(self) -> Tuple[int, int, int, int]:
        """Bounding box axis-aligned (x1, y1, x2, y2) derivado dos corners.

        Útil como aproximação para treino de um detector de placas baseado
        em bounding box (ex: YOLO), já que o dataset original fornece um
        quadrilátero e não um retângulo alinhado aos eixos.
        """
        xs = [c.x for c in self.corners]
        ys = [c.y for c in self.corners]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def vehicle_id(self) -> str:
        """Identificador único do veículo, usado para agrupar splits e
        evitar vazamento de dados (data leakage) entre train/val/test.
        """
        return self.plate
