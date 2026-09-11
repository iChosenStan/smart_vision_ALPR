"""Testes do Módulo 2 do backend — fluxo de entrada (POST /api/entry)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.tests.conftest import make_vehicle_result
from src.pipeline.schemas import FrameResult


def _upload_dummy_image(client: TestClient):
    return client.post("/api/entry", files={"file": ("frame.jpg", b"conteudo-fake-da-imagem", "image/jpeg")})


def test_entry_creates_ticket_when_plate_is_read(make_client) -> None:
    frame_result = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=50.0, fps=20.0)
    with make_client(frame_result) as client:
        response = _upload_dummy_image(client)

    assert response.status_code == 201
    data = response.json()
    assert data["vehicle"]["plate"] == "ABC1234"
    assert data["status"] == "EM_ABERTO"
    assert data["ocr_confidence"] == 0.9
    assert data["entry_image_path"] is not None
    assert data["plate_image_path"] is not None
    assert data["ticket_number"] == f"T{data['id']:06d}"


def test_entry_returns_422_when_no_vehicle_detected(make_client) -> None:
    frame_result = FrameResult(vehicles=[], total_latency_ms=20.0, fps=50.0)
    with make_client(frame_result) as client:
        response = _upload_dummy_image(client)

    assert response.status_code == 422


def test_entry_creates_ticket_with_placeholder_plate_when_ocr_fails(make_client) -> None:
    frame_result = FrameResult(vehicles=[make_vehicle_result(plate=None)], total_latency_ms=40.0, fps=25.0)
    with make_client(frame_result) as client:
        response = _upload_dummy_image(client)

    assert response.status_code == 201
    data = response.json()
    assert data["vehicle"]["plate"].startswith("DESCONHECIDA-")
    assert data["ocr_confidence"] == 0.0
    assert data["plate_image_path"] is None


def test_entry_rejects_empty_file(make_client) -> None:
    frame_result = FrameResult(vehicles=[], total_latency_ms=0.0, fps=0.0)
    with make_client(frame_result) as client:
        response = client.post("/api/entry", files={"file": ("frame.jpg", b"", "image/jpeg")})

    assert response.status_code == 400


def test_entry_reuses_existing_vehicle_on_repeated_plate(make_client) -> None:
    frame_result = FrameResult(vehicles=[make_vehicle_result(plate="XYZ9999")], total_latency_ms=30.0, fps=30.0)
    with make_client(frame_result) as client:
        first = _upload_dummy_image(client)
        second = _upload_dummy_image(client)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["vehicle"]["id"] == second.json()["vehicle"]["id"]
    assert first.json()["id"] != second.json()["id"]


def test_entry_saves_capture_images_to_disk(make_client, tmp_path) -> None:
    frame_result = FrameResult(vehicles=[make_vehicle_result(plate="DEF5678")], total_latency_ms=45.0, fps=22.0)
    with make_client(frame_result) as client:
        response = _upload_dummy_image(client)

    data = response.json()
    vehicle_image = tmp_path / "captures" / data["entry_image_path"]
    plate_image = tmp_path / "captures" / data["plate_image_path"]
    assert vehicle_image.exists()
    assert plate_image.exists()
