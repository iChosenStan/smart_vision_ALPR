"""Endpoint de health-check."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(request: Request) -> dict:
    vision_service = getattr(request.app.state, "vision_service", None)
    models_loaded = vision_service is not None
    return {
        "status": "ok" if models_loaded else "degraded",
        "models_loaded": models_loaded,
    }
