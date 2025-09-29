"""Tests for the FastAPI layer wiring the Rings of Saturn services."""
from __future__ import annotations

import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pytest
import torch

from fastapi.testclient import TestClient
from api.server import app, reset_state


client = TestClient(app)


def setup_function() -> None:
    reset_state()


def test_ledger_endpoints() -> None:
    response = client.post("/ledger/add_tx", json={"sensor": "lumen", "value": 1337})
    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "accepted"
    assert data["pending"] == 1

    response = client.post("/ledger/create_block")
    block_payload = response.json()
    assert block_payload["status"] == "created"
    assert block_payload["block"]["index"] == 0

    response = client.get("/ledger/chain")
    chain = response.json()["chain"]
    assert isinstance(chain, list)
    assert len(chain) == 1


def test_hdag_endpoints() -> None:
    vector = [1.0, 0.0, 0.0]
    other = [0.5, 0.5, 0.0]
    assert client.post("/hdag/add_node", json={"name": "sensor", "vector": vector}).status_code == 200
    assert client.post("/hdag/add_node", json={"name": "feature", "vector": other}).status_code == 200

    edge = client.post("/hdag/add_edge", json={"source": "sensor", "target": "feature", "weight": 0.9})
    assert edge.status_code == 200
    assert edge.json()["edge"]["weight"] == 0.9

    resonance = client.post("/hdag/resonance", json={"source": "sensor", "target": "feature"}).json()
    expected = torch.nn.functional.cosine_similarity(
        torch.tensor(vector), torch.tensor(other), dim=0
    ).item()
    assert pytest.approx(resonance["resonance"], rel=1e-6) == expected


def test_tic_endpoints() -> None:
    payload = {"vectors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    condensed = client.post("/tic/condense", json=payload).json()
    assert condensed["condensed"] == [1.0, 0.0, 0.0]

    invariant = client.post(
        "/tic/invariant", json={"state_a": [1, 2, 3], "state_b": [1.0, 2.0, 3.0]}
    ).json()
    assert invariant["invariant"] is True


def test_ml_endpoints(monkeypatch) -> None:
    calls: dict[str, int] = {"train": 0}

    def fake_train(config: Any) -> None:  # pragma: no cover - executed in tests
        calls["train"] += 1

    monkeypatch.setattr("ml.demo_training.train", fake_train)

    prediction = client.post("/ml/predict", json={"vector": [0.1, 0.2, 0.3]}).json()
    assert pytest.approx(prediction["prediction"], rel=1e-6) == 0.2

    training = client.post("/ml/train_demo").json()
    assert training["status"] in {"completed", "skipped"}
    if training["status"] == "completed":
        assert calls["train"] == 1


def test_zkml_endpoints() -> None:
    inference = client.post("/zkml/zk_infer", json={"vector": [0.4, 0.4]}).json()
    assert inference["proof"].startswith("ZK-PROOF:")
    assert "statement" in inference

    verification = client.post(
        "/zkml/verify", json={"statement": inference["statement"], "proof": inference["proof"]}
    ).json()
    assert verification["valid"] is True

