"""A lightweight torch compatibility layer for testing purposes."""
from __future__ import annotations

import math
from typing import Sequence, Union


Number = Union[int, float]
TensorInput = Union[Number, Sequence[Number], "Tensor"]


class Tensor:
    """Minimal Tensor implementation supporting required operations."""

    def __init__(self, data: TensorInput) -> None:
        if isinstance(data, Tensor):
            self._values = data._values[:]
        elif isinstance(data, (int, float)):
            self._values = [float(data)]
        elif isinstance(data, Sequence):
            self._values = [float(v) for v in data]
        else:
            raise TypeError("Unsupported data type for Tensor")

    def norm(self) -> "Tensor":
        return Tensor(math.sqrt(sum(v * v for v in self._values)))

    def flatten(self) -> "Tensor":
        return Tensor(self)

    def item(self) -> float:
        if len(self._values) != 1:
            raise ValueError("Can only convert a single element tensor to a Python scalar")
        return self._values[0]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Tensor({self._values!r})"


def tensor(data: TensorInput) -> Tensor:
    return Tensor(data)


def dot(x: Tensor, y: Tensor) -> Tensor:
    x_vals = x.flatten()._values
    y_vals = y.flatten()._values
    if len(x_vals) != len(y_vals):
        raise ValueError("Dot product requires tensors of the same length")
    return Tensor(sum(a * b for a, b in zip(x_vals, y_vals)))


def equal(x: Tensor, y: Tensor) -> bool:
    return x.flatten()._values == y.flatten()._values


__all__ = ["Tensor", "tensor", "dot", "equal"]
