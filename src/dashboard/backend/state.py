"""Mutable dashboard state shared across API endpoints."""
from __future__ import annotations

from typing import Any, Dict

_DEFAULT_PROOF: Dict[str, Any] = {
    "input": [],
    "prediction": None,
    "proof": "",
    "statement": "",
    "verified": False,
}

_last_proof: Dict[str, Any] = dict(_DEFAULT_PROOF)


def set_last_proof(proof: Dict[str, Any]) -> None:
    """Store the latest ZKML proof result returned by the pipeline."""

    global _last_proof
    # Normalise to a JSON-friendly payload and keep only known keys so we don't
    # accidentally leak large tensors to the frontend consumers.
    cleaned: Dict[str, Any] = dict(_DEFAULT_PROOF)
    cleaned.update({k: proof.get(k) for k in cleaned.keys() if k in proof})
    # Ensure inputs are converted to plain Python lists for serialisation.
    vector = proof.get("input") or proof.get("vector")
    if vector is not None:
        if hasattr(vector, "tolist"):
            cleaned["input"] = list(vector.tolist())
        elif isinstance(vector, (list, tuple)):
            cleaned["input"] = [float(v) for v in vector]
    prediction = proof.get("prediction")
    if hasattr(prediction, "item"):
        try:
            prediction = float(prediction.item())
        except Exception:  # pragma: no cover - defensive conversion
            prediction = float(prediction)
    if isinstance(prediction, (int, float)):
        cleaned["prediction"] = float(prediction)
    cleaned["proof"] = str(proof.get("proof", cleaned["proof"]))
    cleaned["statement"] = str(proof.get("statement", cleaned["statement"]))
    cleaned["verified"] = bool(proof.get("verified", cleaned["verified"]))
    _last_proof = cleaned


def get_last_proof() -> Dict[str, Any]:
    """Return the latest proof payload served to the dashboard."""

    return dict(_last_proof)


def reset_last_proof() -> None:
    """Reset the stored proof to its default empty state."""

    global _last_proof
    _last_proof = dict(_DEFAULT_PROOF)


__all__ = ["get_last_proof", "reset_last_proof", "set_last_proof"]
