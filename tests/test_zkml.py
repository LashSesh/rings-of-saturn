"""Tests for the mock zero-knowledge machine learning utilities."""

from src.zkml import ZKML


def test_zk_inference_returns_output_and_proof():
    zkml = ZKML()

    def model(x):
        return x * 2

    x = 5
    y, proof = zkml.zk_inference(model, x)

    assert y == 10
    assert proof == zkml.PROOF_PLACEHOLDER


def test_verify_inference_accepts_valid_proof():
    zkml = ZKML()

    x = 3
    y = 6
    proof = zkml.PROOF_PLACEHOLDER

    assert zkml.verify_inference(proof, x, y)
