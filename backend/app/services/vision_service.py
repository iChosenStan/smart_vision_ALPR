"""Serviço que encapsula o `SmartVisionPipeline` (Etapa 7) para uso pela API.

Não reimplementa NENHUMA lógica de IA — é uma camada fina de adaptação
entre o pipeline de visão computacional já treinado e testado
(`src/pipeline`) e o contexto de uma requisição HTTP (bytes de imagem →
resultado estruturado). Os modelos são carregados uma única vez, no
startup da aplicação (ver `main.py`), nunca por requisição.

NOTA DE DESIGN: o import de `SmartVisionPipeline` é feito dentro de
`load()` (import tardio), não no topo do módulo. Isso evita forçar a
instalação de `torch`/`ultralytics`/`paddleocr` só para importar este
arquivo — útil para testes que substituem `VisionService` por um fake
(não tocam a IA de verdade) e para qualquer contexto que só precise da
API sem rodar inferência real.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.core.logging import get_logger

if TYPE_CHECKING:
    from src.pipeline.schemas import FrameResult
    from src.pipeline.smart_vision_pipeline import SmartVisionPipeline

logger = get_logger(__name__)


class VisionService:
    """Wrapper fino sobre `SmartVisionPipeline`."""

    def __init__(self, pipeline: "SmartVisionPipeline") -> None:
        self._pipeline = pipeline

    @classmethod
    def load(cls, config_path: Union[str, Path, None] = None) -> "VisionService":
        """Carrega os 4 modelos reais treinados (Etapas 3-6) via o pipeline
        integrado da Etapa 7. Chamado uma única vez, no startup do FastAPI.
        """
        from src.pipeline.smart_vision_pipeline import SmartVisionPipeline

        path = config_path or settings.pipeline_config_path
        logger.info(f"Carregando SmartVisionPipeline a partir de {path}...")
        pipeline = SmartVisionPipeline.from_config(path)
        logger.info("Pipeline de IA carregado com sucesso (modelos reais, não fictícios).")
        return cls(pipeline)

    def process_image_bytes(self, image_bytes: bytes) -> "FrameResult":
        """Decodifica bytes de imagem (upload HTTP ou frame de webcam/vídeo)
        e roda o pipeline completo: veículo → placa → OCR → classificação.

        Raises:
            ValueError: Se os bytes não puderem ser decodificados como imagem.
        """
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Não foi possível decodificar a imagem enviada.")
        return self._pipeline.process_frame(frame)
