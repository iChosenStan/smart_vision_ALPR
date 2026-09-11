"""Logging centralizado do SmartVision ALPR."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger

_CONFIGURED = False


def _configure_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "outputs/logs",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        colorize=True,
    )
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_path / "smartvision.log",
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None):
    _configure_logger()
    return _logger.bind(name=name or "smartvision")
