"""Integration tests for the dashboard FastAPI router."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.server import app, reset_state

client = TestClient(app)


def setup_function() -> None:
    reset_state()


def test_get_ledger() -> None:
    response = client.get("/dashboard/ledger")
    assert response.status_code == 200
    payload = response.json()
    assert "chain" in payload and isinstance(payload["chain"], list)
    assert "pending" in payload and isinstance(payload["pending"], list)


def test_get_hdag() -> None:
    response = client.get("/dashboard/hdag")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"nodes", "edges"}
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)


def test_get_spiral() -> None:
    response = client.get("/dashboard/spiral?n=5&a=1&b=0.5&c=0.1")
    assert response.status_code == 200
    payload = response.json()
    assert "points" in payload
    assert len(payload["points"]) == 5
    first_point = payload["points"][0]
    assert "coordinates" in first_point and len(first_point["coordinates"]) == 5


def test_get_tic_state() -> None:
    response = client.get("/dashboard/tic")
    assert response.status_code == 200
    payload = response.json()
    assert "history" in payload and isinstance(payload["history"], list)


def test_ml_status() -> None:
    response = client.get("/dashboard/ml/train_status")
    assert response.status_code == 200
    payload = response.json()
    assert "accuracy" in payload and len(payload["accuracy"]) == len(payload["epochs"])


def test_get_latest_proof_updates_on_infer() -> None:
    # Ensure inference populates the shared proof state
    infer_response = client.post("/zkml/zk_infer", json={"vector": [1.0, 2.0, 3.0]})
    assert infer_response.status_code == 200
    proof_response = client.get("/dashboard/zkml/proof")
    assert proof_response.status_code == 200
    proof_payload = proof_response.json()
    assert proof_payload["input"] == [1.0, 2.0, 3.0]
    assert proof_payload["proof"]
