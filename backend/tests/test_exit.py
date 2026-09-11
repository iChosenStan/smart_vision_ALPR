"""Testes do Módulo 3 do backend — fluxo de saída (POST /api/exit)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.tests.conftest import make_vehicle_result
from src.pipeline.schemas import FrameResult


def _upload(client: TestClient, path: str):
    return client.post(path, files={"file": ("frame.jpg", b"fake", "image/jpeg")})


def test_exit_opens_gate_when_ticket_is_paid(make_client_sequence) -> None:
    entry_frame = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    exit_frame = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)

    with make_client_sequence([entry_frame, exit_frame]) as client:
        ticket = _upload(client, "/api/entry").json()
        client.put(f"/api/tickets/{ticket['id']}/pay", json={})
        response = _upload(client, "/api/exit")

    assert response.status_code == 200
    data = response.json()
    assert data["gate_open"] is True
    assert data["ticket"]["status"] == "FINALIZADO"
    assert data["ticket"]["exit_at"] is not None


def test_exit_denies_gate_when_payment_pending(make_client_sequence) -> None:
    entry_frame = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    exit_frame = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)

    with make_client_sequence([entry_frame, exit_frame]) as client:
        _upload(client, "/api/entry")
        response = _upload(client, "/api/exit")  # sem pagar

    assert response.status_code == 200
    data = response.json()
    assert data["gate_open"] is False
    assert "pendente" in data["message"].lower()
    assert data["ticket"]["status"] == "EM_ABERTO"


def test_exit_returns_404_for_unknown_plate(make_client_sequence) -> None:
    exit_frame = FrameResult(vehicles=[make_vehicle_result(plate="NUNCA0001")], total_latency_ms=1.0, fps=1.0)

    with make_client_sequence([exit_frame]) as client:
        response = _upload(client, "/api/exit")

    assert response.status_code == 404


def test_exit_returns_422_when_no_vehicle_detected(make_client_sequence) -> None:
    exit_frame = FrameResult(vehicles=[], total_latency_ms=1.0, fps=1.0)

    with make_client_sequence([exit_frame]) as client:
        response = _upload(client, "/api/exit")

    assert response.status_code == 422


def test_exit_returns_404_when_vehicle_already_exited(make_client_sequence) -> None:
    entry_frame = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    exit_frame_1 = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    exit_frame_2 = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)

    with make_client_sequence([entry_frame, exit_frame_1, exit_frame_2]) as client:
        ticket = _upload(client, "/api/entry").json()
        client.put(f"/api/tickets/{ticket['id']}/pay", json={})
        _upload(client, "/api/exit")  # 1ª saída — sucesso, ticket finalizado
        response = _upload(client, "/api/exit")  # 2ª tentativa — não há mais ticket aberto

    assert response.status_code == 404
