"""Tests for the zero-knowledge machine learning helpers."""
from __future__ import annotations

import random

import pytest

try:  # pragma: no cover - ensure local torch stub is importable
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import torch  # type: ignore

from src.zkml import build_statement, build_witness, generate_proof, verify_proof, zk_infer


class TinyLinear(torch.nn.Module):
    """Minimal linear model with manual gradient descent."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = random.uniform(-0.5, 0.5)
        self.bias = random.uniform(-0.5, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        values = x.flatten().tolist()
        outputs = [self.weight * v + self.bias for v in values]
        return torch.tensor(outputs)


def _train_tiny_model() -> TinyLinear:
    random.seed(0)
    model = TinyLinear()
    lr = 0.05
    samples = [
        (-1.0, -3.25),
        (-0.5, -2.0),
        (0.0, -0.75),
        (0.5, 0.5),
        (1.0, 1.75),
    ]

    for _ in range(250):
        grad_w = 0.0
        grad_b = 0.0
        for x_val, y_val in samples:
            pred = model.weight * x_val + model.bias
            error = pred - y_val
            grad_w += error * x_val
            grad_b += error
        grad_w /= len(samples)
        grad_b /= len(samples)
        model.weight -= lr * grad_w
        model.bias -= lr * grad_b

    return model


def test_zk_infer_generates_verifiable_proof() -> None:
    model = _train_tiny_model()
    x_eval = torch.tensor([0.25])

    prediction, proof = zk_infer(model, x_eval)

    witness = build_witness(model, x_eval)
    statement = build_statement(prediction, witness)

    assert proof.startswith("ZK-PROOF:")
    assert verify_proof(statement, proof)


def test_generate_proof_rejects_mismatched_witness() -> None:
    model = _train_tiny_model()
    x_eval = torch.tensor([0.1])
    prediction, proof = zk_infer(model, x_eval)

    witness = build_witness(model, x_eval)
    statement = build_statement(prediction, witness)

    bad_witness = dict(witness)
    bad_witness["model"] = "TamperedModel"

    with pytest.raises(ValueError, match="commitment"):
        generate_proof(statement, bad_witness)

    assert verify_proof(statement, proof)


def test_verify_proof_detects_statement_tampering() -> None:
    model = _train_tiny_model()
    x_eval = torch.tensor([0.0])
    prediction, proof = zk_infer(model, x_eval)

    witness = build_witness(model, x_eval)
    statement = build_statement(prediction, witness)

    tampered = statement.replace("commitment", "commitment_mod")
    assert not verify_proof(tampered, proof)
