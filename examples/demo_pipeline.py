"""Demonstration pipeline that connects all core components.

The pipeline follows the Rings of Saturn processing stages:

1. A transaction is added to the ledger and materialised into a block.
2. The block is converted into a tensor and inserted into the HDAG.
3. The tensor is projected into a spiral point representation.
4. Spiral points are condensed into a TIC (Temporal Information Crystal).

The :func:`process_transaction` helper performs these steps and returns the
final TIC state for convenient inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import hashlib
import math

import torch

from src.hdag.hdag import HDAG
from src.ledger import Ledger


@dataclass
class SpiralPoint:
    """Representation of a point on the spiral manifold.

    The coordinates are deliberately lightweight aggregates derived from a
    tensor. They are sufficient for demonstration purposes while keeping the
    tests deterministic.
    """

    radius: float
    angle: float
    height: float

    def to_dict(self) -> Dict[str, float]:
        """Serialise the point to a dictionary for downstream usage."""

        return {"radius": self.radius, "angle": self.angle, "height": self.height}


class TIC:
    """Temporal Information Crystal that condenses spiral points."""

    def __init__(self) -> None:
        self._points: List[SpiralPoint] = []

    def condense(self, point: SpiralPoint) -> Dict[str, Any]:
        """Add a spiral point and return the updated TIC state."""

        self._points.append(point)
        return self.as_dict()

    def _centroid(self) -> Dict[str, float]:
        if not self._points:
            return {"radius": 0.0, "angle": 0.0, "height": 0.0}
        radius = sum(p.radius for p in self._points) / len(self._points)
        angle = sum(p.angle for p in self._points) / len(self._points)
        height = sum(p.height for p in self._points) / len(self._points)
        return {"radius": radius, "angle": angle, "height": height}

    def as_dict(self) -> Dict[str, Any]:
        """Return the full TIC state as a dictionary."""

        return {
            "points": [point.to_dict() for point in self._points],
            "count": len(self._points),
            "centroid": self._centroid(),
        }


def _hash_to_scalar(value: str) -> float:
    """Map a hexadecimal string to a stable floating point scalar in [0, 1)."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _block_to_tensor(block: Dict[str, Any]) -> torch.Tensor:
    """Convert a ledger block into a compact tensor representation."""

    transactions: Iterable[Any] = block.get("transactions", [])
    features = torch.tensor(
        [
            float(block.get("index", 0)),
            float(block.get("timestamp", 0.0)),
            float(len(list(transactions))),
            _hash_to_scalar(str(block.get("prev_hash", ""))),
            _hash_to_scalar(str(block.get("hash", ""))),
        ]
    )
    return features


def _tensor_to_spiral_point(tensor: torch.Tensor) -> SpiralPoint:
    """Project a tensor into the spiral coordinate system."""

    values = tensor.flatten()._values
    radius = tensor.norm().item()
    # Offset denominators slightly to avoid divide-by-zero in degenerate cases.
    angle = math.atan2(values[-1], values[0] + 1e-6)
    height = sum(values) / len(values)
    return SpiralPoint(radius=radius, angle=angle, height=height)


def process_transaction(tx: Any) -> Dict[str, Any]:
    """Run a transaction through the complete Rings of Saturn pipeline."""

    ledger = Ledger()
    hdag = HDAG()
    tic = TIC()

    ledger.add_transaction(tx)
    block = ledger.create_block()

    tensor = _block_to_tensor(block)
    node_id = block["hash"]
    hdag.add_node(node_id, tensor)

    spiral_point = _tensor_to_spiral_point(tensor)
    tic_state = tic.condense(spiral_point)

    return tic_state


__all__ = ["process_transaction", "SpiralPoint", "TIC"]

