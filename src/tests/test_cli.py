"""Tests for the Typer based Rings of Saturn CLI."""
from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pytest

from typer.testing import CliRunner
from api.server import reset_state
from cli.main import app


runner = CliRunner()


def setup_function() -> None:
    reset_state()


def _invoke(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.stdout
    return result.stdout.strip()


def test_ledger_commands() -> None:
    response = json.loads(_invoke("ledger", "add-tx", '{"sensor":"temp","value":42}'))
    assert response["status"] == "accepted"

    block = json.loads(_invoke("ledger", "create-block"))
    assert block["block"]["index"] == 0

    chain = json.loads(_invoke("ledger", "show"))
    assert len(chain["chain"]) == 1


def test_hdag_commands() -> None:
    json.loads(_invoke("hdag", "add-node", "sensor", "[1,0,0]"))
    json.loads(_invoke("hdag", "add-node", "feature", "[1,0,0]"))

    edge = json.loads(_invoke("hdag", "add-edge", "sensor", "feature", "0.9"))
    assert edge["edge"]["weight"] == 0.9

    resonance = json.loads(_invoke("hdag", "resonance", "sensor", "feature"))
    assert pytest.approx(resonance["resonance"], rel=1e-6) == 1.0


def test_tic_commands(tmp_path: Path) -> None:
    condensed = json.loads(_invoke("tic", "condense", "[[1,0,0],[0,1,0]]"))
    assert condensed["condensed"] == [1.0, 0.0, 0.0]

    state_a = tmp_path / "tic_a.json"
    state_b = tmp_path / "tic_b.json"
    state_a.write_text(json.dumps([1, 2, 3]))
    state_b.write_text(json.dumps([1, 2, 3]))

    invariant = json.loads(_invoke("tic", "invariant", str(state_a), str(state_b)))
    assert invariant["invariant"] is True


def test_ml_commands(monkeypatch) -> None:
    calls: dict[str, int] = {"train": 0}

    def fake_train(config: Any) -> None:  # pragma: no cover - executed in tests
        calls["train"] += 1

    monkeypatch.setattr("ml.demo_training.train", fake_train)

    prediction = json.loads(_invoke("ml", "predict", "[0.1,0.2,0.3]"))
    assert pytest.approx(prediction["prediction"], rel=1e-6) == 0.2

    training = json.loads(_invoke("ml", "train-demo"))
    assert training["status"] in {"completed", "skipped"}
    if training["status"] == "completed":
        assert calls["train"] == 1


def test_zkml_commands() -> None:
    inference = json.loads(_invoke("zkml", "zk-infer", "[0.5,0.1,0.9]"))
    assert inference["proof"].startswith("ZK-PROOF:")

    verification = json.loads(_invoke("zkml", "verify", inference["statement"], inference["proof"]))
    assert verification["valid"] is True

