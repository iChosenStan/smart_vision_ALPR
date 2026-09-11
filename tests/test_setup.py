"""Testes da Etapa 1 — estrutura base do projeto.

Valida:
    - Carregamento do arquivo de configuração base.
    - Funcionamento do logger centralizado.
    - Hierarquia de exceções customizadas.
"""

from pathlib import Path

import yaml

from src.utils.exceptions import (
    ClassificationError,
    OCRError,
    SmartVisionError,
)
from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_base_config_exists_and_is_valid_yaml() -> None:
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    assert config_path.exists(), "configs/base.yaml não encontrado"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["project"]["name"] == "SmartVision ALPR"
    assert config["dataset"]["name"] == "UFPR-VeSV"
    assert config["pipeline_targets"]["ocr_accuracy_min"] == 0.95


def test_logger_returns_usable_instance() -> None:
    logger = get_logger(__name__)
    # Não deve lançar exceção ao logar em diferentes níveis
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")


def test_custom_exceptions_inherit_from_base() -> None:
    assert issubclass(OCRError, SmartVisionError)
    assert issubclass(ClassificationError, SmartVisionError)

    try:
        raise OCRError("falha simulada de OCR")
    except SmartVisionError as exc:
        assert "falha simulada de OCR" in str(exc)


def test_expected_directory_structure_exists() -> None:
    expected_dirs = [
        "configs",
        "datasets/raw",
        "datasets/processed",
        "models/pretrained",
        "models/checkpoints",
        "outputs/logs",
        "outputs/predictions",
        "outputs/reports",
        "src/detection",
        "src/alpr",
        "src/classification",
        "src/preprocessing",
        "src/training",
        "src/evaluation",
        "src/visualization",
        "src/utils",
    ]
    for rel_dir in expected_dirs:
        assert (PROJECT_ROOT / rel_dir).is_dir(), f"Diretório ausente: {rel_dir}"
