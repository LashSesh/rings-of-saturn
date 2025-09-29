"""Mocked zero-knowledge proof generation and verification utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

_PROOF_PREFIX = "ZK-PROOF:"


def _hash_witness(witness: Mapping[str, Any]) -> str:
    """Create a SHA256 commitment for a witness mapping."""
    witness_payload = json.dumps(witness, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(witness_payload.encode("utf-8")).hexdigest()


def generate_proof(statement: str, witness: Mapping[str, Any]) -> str:
    """Generate a placeholder zero-knowledge proof for a statement.

    Parameters
    ----------
    statement:
        Canonical string representation of the public statement that should be
        proven. For ZKML inference this typically includes a prediction and a
        commitment to the witness.
    witness:
        Private data used to create the proof. The mapping is hashed to produce
        a commitment that is expected to be embedded in the ``statement``.

    Returns
    -------
    str
        Placeholder proof string in the form ``"ZK-PROOF:<hash>"`` where the
        hash is computed with SHA256 over the provided ``statement``.

    Raises
    ------
    ValueError
        If the witness commitment embedded in ``statement`` does not match the
        supplied ``witness`` mapping.
    """

    commitment = _hash_witness(witness)
    try:
        payload = json.loads(statement)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and "commitment" in payload:
        if payload["commitment"] != commitment:
            raise ValueError("Witness commitment does not match statement.")

    statement_hash = hashlib.sha256(statement.encode("utf-8")).hexdigest()
    return f"{_PROOF_PREFIX}{statement_hash}"


def verify_proof(statement: str, proof: str) -> bool:
    """Verify a placeholder zero-knowledge proof for a statement."""
    if not proof.startswith(_PROOF_PREFIX):
        return False

    proof_hash = proof[len(_PROOF_PREFIX) :]
    expected_hash = hashlib.sha256(statement.encode("utf-8")).hexdigest()
    return proof_hash == expected_hash


__all__ = ["generate_proof", "verify_proof"]
