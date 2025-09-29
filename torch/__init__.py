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

    def tolist(self) -> list[float]:
        return self._values[:]

    def detach(self) -> "Tensor":
        return Tensor(self)

    def cpu(self) -> "Tensor":
        return Tensor(self)

    def clone(self) -> "Tensor":
        return Tensor(self)

    def __iter__(self):  # pragma: no cover - trivial iterator
        return iter(self._values)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._values)

    def __getitem__(self, item):  # pragma: no cover - trivial
        return self._values[item]

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


class Module:
    """Minimal drop-in replacement for :class:`torch.nn.Module`."""

    def forward(self, *args, **kwargs):  # pragma: no cover - override required
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _NNNamespace:
    Module = Module


nn = _NNNamespace()


__all__ = ["Tensor", "tensor", "dot", "equal", "nn", "Module"]
