"""Configuração do banco de dados (SQLAlchemy) — SQLite para o PoC."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Classe base declarativa para todos os modelos ORM do backend."""


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        _engine = create_engine(settings.database_url, connect_args=connect_args)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        logger.info(f"Engine de banco de dados criada: {settings.database_url}")
    return _engine


def get_db():
    _get_engine()
    assert _SessionLocal is not None
    db: Session = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from backend.app import models  # noqa: F401

    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Schema do banco de dados verificado/criado.")


def reset_engine_for_testing() -> None:
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
