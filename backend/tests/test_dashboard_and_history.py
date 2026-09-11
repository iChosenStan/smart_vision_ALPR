"""Testes do Módulo 4 do backend — dashboard, listagem e histórico."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.tests.conftest import make_vehicle_result
from src.pipeline.schemas import FrameResult


def _upload(client: TestClient, path: str):
    return client.post(path, files={"file": ("frame.jpg", b"fake", "image/jpeg")})


def test_dashboard_reflects_created_tickets(make_client_sequence) -> None:
    frames = [
        FrameResult(vehicles=[make_vehicle_result(plate="AAA1111")], total_latency_ms=1.0, fps=1.0),
        FrameResult(vehicles=[make_vehicle_result(plate="BBB2222")], total_latency_ms=1.0, fps=1.0),
    ]
    with make_client_sequence(frames) as client:
        t1 = _upload(client, "/api/entry").json()
        _upload(client, "/api/entry")
        client.put(f"/api/tickets/{t1['id']}/pay", json={})

        response = client.get("/api/dashboard")

    assert response.status_code == 200
    data = response.json()
    assert data["total_vehicles"] == 2
    assert data["vehicles_parked"] == 2  # nenhum saiu ainda
    assert data["tickets_paid"] == 1
    assert data["tickets_pending"] == 1
    assert data["simulated_revenue"] == 5.0
    assert data["available_spots"] == data["total_spots"] - 2


def test_dashboard_available_spots_never_negative(make_client_sequence) -> None:
    frame = FrameResult(vehicles=[], total_latency_ms=1.0, fps=1.0)
    with make_client_sequence([frame]) as client:
        response = client.get("/api/dashboard")

    data = response.json()
    assert data["available_spots"] == data["total_spots"]


def test_list_tickets_returns_all_by_default(make_client_sequence) -> None:
    frames = [
        FrameResult(vehicles=[make_vehicle_result(plate="AAA1111")], total_latency_ms=1.0, fps=1.0),
        FrameResult(vehicles=[make_vehicle_result(plate="BBB2222")], total_latency_ms=1.0, fps=1.0),
    ]
    with make_client_sequence(frames) as client:
        _upload(client, "/api/entry")
        _upload(client, "/api/entry")
        response = client.get("/api/tickets")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2
    assert data["page"] == 1


def test_list_tickets_filters_by_plate(make_client_sequence) -> None:
    frames = [
        FrameResult(vehicles=[make_vehicle_result(plate="AAA1111")], total_latency_ms=1.0, fps=1.0),
        FrameResult(vehicles=[make_vehicle_result(plate="BBB2222")], total_latency_ms=1.0, fps=1.0),
    ]
    with make_client_sequence(frames) as client:
        _upload(client, "/api/entry")
        _upload(client, "/api/entry")
        response = client.get("/api/tickets", params={"plate": "AAA"})

    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["vehicle"]["plate"] == "AAA1111"


def test_list_tickets_filters_by_status(make_client_sequence) -> None:
    frames = [
        FrameResult(vehicles=[make_vehicle_result(plate="AAA1111")], total_latency_ms=1.0, fps=1.0),
        FrameResult(vehicles=[make_vehicle_result(plate="BBB2222")], total_latency_ms=1.0, fps=1.0),
    ]
    with make_client_sequence(frames) as client:
        t1 = _upload(client, "/api/entry").json()
        _upload(client, "/api/entry")
        client.put(f"/api/tickets/{t1['id']}/pay", json={})

        response = client.get("/api/tickets", params={"status": "PAGO"})

    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["status"] == "PAGO"


def test_list_tickets_pagination(make_client_sequence) -> None:
    frames = [
        FrameResult(vehicles=[make_vehicle_result(plate=f"AAA{i:04d}")], total_latency_ms=1.0, fps=1.0)
        for i in range(5)
    ]
    with make_client_sequence(frames) as client:
        for _ in range(5):
            _upload(client, "/api/entry")

        page1 = client.get("/api/tickets", params={"page": 1, "page_size": 2}).json()
        page2 = client.get("/api/tickets", params={"page": 2, "page_size": 2}).json()

    assert page1["total"] == 5
    assert len(page1["items"]) == 2
    assert len(page2["items"]) == 2
    assert page1["items"][0]["id"] != page2["items"][0]["id"]


def test_history_endpoint_returns_same_shape_as_tickets_list(make_client_sequence) -> None:
    frame = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    with make_client_sequence([frame]) as client:
        _upload(client, "/api/entry")
        response = client.get("/api/history")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data and "total" in data and "page" in data
    assert data["total"] == 1


def test_ticket_read_exposes_duration_minutes(make_client_sequence) -> None:
    frame = FrameResult(vehicles=[make_vehicle_result(plate="ABC1234")], total_latency_ms=1.0, fps=1.0)
    with make_client_sequence([frame]) as client:
        response = _upload(client, "/api/entry")

    assert response.json()["duration_minutes"] >= 0
