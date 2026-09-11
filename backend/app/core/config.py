"""Configuração da aplicação backend, carregada via variáveis de ambiente (.env)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "SmartVision ALPR — Estacionamento Inteligente"
    database_url: str = f"sqlite:///{_PROJECT_ROOT / 'database' / 'smartvision.db'}"
    pipeline_config_path: Path = _PROJECT_ROOT / "configs" / "pipeline.yaml"
    captures_dir: Path = _PROJECT_ROOT / "backend" / "captures"
    cors_origins: List[str] = ["http://localhost:3000"]
    parking_rate_per_hour: float = 5.0
    total_parking_spots: int = 50


settings = Settings()
