"""FastAPI router powering the Rings of Saturn dashboard endpoints."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import APIRouter, Query

from spiral.spiral import Spiral

from .services import hdag_service, ledger_service, tic_service
from .state import get_last_proof

router = APIRouter(tags=["dashboard"])


def _serialise_ledger() -> Dict[str, List[Dict[str, object]]]:
    """Return a serialisable snapshot of the ledger state."""

    chain = []
    for block in ledger_service.to_dict().get("chain", []):
        block_copy = dict(block)
        projection = block_copy.get("projection")
        if hasattr(projection, "tolist"):
            block_copy["projection"] = list(projection.tolist())
        chain.append(block_copy)
    pending = [dict(tx) for tx in getattr(ledger_service, "pending_transactions", [])]
    return {"chain": chain, "pending": pending}


def _serialise_hdag() -> Dict[str, object]:
    """Return nodes and edges from the HDAG service."""

    nodes = [
        {"id": name, "vector": hdag_service._tensor_to_list(vector)}
        for name, vector in hdag_service.items()
    ]
    edges = [
        {"source": src, "target": dst, "weight": float(weight)}
        for src, dst, weight in getattr(hdag_service, "edges", [])
    ]
    return {"nodes": nodes, "edges": edges}


def _tic_history() -> List[List[float]]:
    """Generate a small deterministic TIC history for visualisation."""

    base = [
        [math.cos(idx), math.sin(idx), math.cos(idx) * math.sin(idx)]
        for idx in [0.0, 0.7, 1.4, 2.1, 2.8, 3.5]
    ]
    if tic_service.state is not None:
        base.append(tic_service._to_flat_list(tic_service.state))
    return base


@router.get("/ledger")
def get_ledger() -> Dict[str, object]:
    """Return the current ledger chain and pending transactions."""

    return _serialise_ledger()


@router.get("/hdag")
def get_hdag() -> Dict[str, object]:
    """Return nodes and edges used to build the HDAG visualisation."""

    return _serialise_hdag()


@router.get("/spiral")
def get_spiral(
    n: int = Query(100, ge=1, le=2000, description="Number of spiral points"),
    a: float = Query(1.0, description="First radial coefficient"),
    b: float = Query(0.5, description="Second radial coefficient"),
    c: float = Query(0.1, description="Axial growth coefficient"),
) -> Dict[str, object]:
    """Return ``n`` points sampled from the configured spiral."""

    spiral = Spiral(a=a, b=b, c=c)
    thetas = [idx * (4 * math.pi) / max(n - 1, 1) for idx in range(n)]
    points = []
    for theta in thetas:
        vector = spiral.map(theta).tolist()
        points.append({
            "theta": theta,
            "coordinates": vector,
        })
    return {"points": points, "params": {"a": a, "b": b, "c": c}}


@router.get("/tic")
def get_tic_state() -> Dict[str, object]:
    """Return the condensed TIC state and a deterministic history."""

    tic_state = tic_service.to_dict().get("tic")
    return {
        "state": tic_state,
        "history": _tic_history(),
    }


@router.get("/tic/active")
def get_tic_active_vectors() -> Dict[str, object]:
    """Return the raw vectors currently stored inside the TIC condensate."""

    if tic_service.state is None:
        return {"vectors": []}
    return {"vectors": [tic_service._to_flat_list(tic_service.state)]}


@router.get("/ml/train_status")
def get_ml_train_status() -> Dict[str, object]:
    """Return a dummy training curve for the ML view."""

    epochs = list(range(1, 11))
    start = datetime.utcnow() - timedelta(minutes=len(epochs))
    timeline = [(start + timedelta(minutes=idx)).isoformat() for idx in range(len(epochs))]
    accuracy = [round(0.5 + 0.05 * math.log(idx + 1), 4) for idx in range(len(epochs))]
    loss = [round(1.2 / (idx + 1), 4) for idx in range(len(epochs))]
    return {"epochs": epochs, "timeline": timeline, "accuracy": accuracy, "loss": loss}


@router.get("/zkml/proof")
def get_latest_proof() -> Dict[str, object]:
    """Return the last ZKML proof produced through the API."""

    return get_last_proof()
