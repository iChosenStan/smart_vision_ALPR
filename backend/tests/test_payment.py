"""Testes do Módulo 3 do backend — pagamento (PUT /api/tickets/{id}/pay)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.tests.conftest import make_vehicle_result
from src.pipeline.schemas import FrameResult


def _create_ticket(client: TestClient, plate: str = "ABC1234") -> dict:
    response = client.post("/api/entry", files={"file": ("frame.jpg", b"fake", "image/jpeg")})
    assert response.status_code == 201
    return response.json()


def test_pay_ticket_marks_as_paid_and_calculates_amount(make_client) -> None:
    frame_result = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    with make_client(frame_result) as client:
        ticket = _create_ticket(client)
        response = client.put(f"/api/tickets/{ticket['id']}/pay", json={"payment_method": "pix"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PAGO"
    assert data["payment_method"] == "pix"
    assert data["amount"] == 5.0  # 1h mínima * R$5.0/h (default)
    assert data["paid_at"] is not None


def test_pay_ticket_uses_default_payment_method(make_client) -> None:
    frame_result = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    with make_client(frame_result) as client:
        ticket = _create_ticket(client)
        response = client.put(f"/api/tickets/{ticket['id']}/pay", json={})

    assert response.status_code == 200
    assert response.json()["payment_method"] == "dinheiro"


def test_pay_ticket_twice_returns_409(make_client) -> None:
    frame_result = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    with make_client(frame_result) as client:
        ticket = _create_ticket(client)
        client.put(f"/api/tickets/{ticket['id']}/pay", json={})
        second_payment = client.put(f"/api/tickets/{ticket['id']}/pay", json={})

    assert second_payment.status_code == 409


def test_pay_nonexistent_ticket_returns_404(make_client) -> None:
    frame_result = FrameResult(vehicles=[], total_latency_ms=1.0, fps=1.0)
    with make_client(frame_result) as client:
        response = client.put("/api/tickets/9999/pay", json={})

    assert response.status_code == 404


def test_get_ticket_by_id(make_client) -> None:
    frame_result = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    with make_client(frame_result) as client:
        ticket = _create_ticket(client)
        response = client.get(f"/api/tickets/{ticket['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == ticket["id"]


def test_get_nonexistent_ticket_returns_404(make_client) -> None:
    frame_result = FrameResult(vehicles=[], total_latency_ms=1.0, fps=1.0)
    with make_client(frame_result) as client:
        response = client.get("/api/tickets/9999")

    assert response.status_code == 404
