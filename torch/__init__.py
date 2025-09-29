"""A lightweight torch compatibility layer for testing purposes."""
from __future__ import annotations

import contextlib
import math
import sys
import types
from typing import Iterable, Sequence, Union


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


def tensor(data: TensorInput, dtype: object | None = None) -> Tensor:
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


class _Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self._modules = modules

    def forward(self, x):  # pragma: no cover - trivial passthrough
        result = x
        for module in self._modules:
            result = module(result)
        return result


class _Passthrough(Module):
    def forward(self, x):  # pragma: no cover - trivial passthrough
        return Tensor(x)


class _ReLU(Module):
    def forward(self, x):
        values = Tensor(x).tolist()
        return Tensor([max(v, 0.0) for v in values])


class _Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.out_features = out_features

    def forward(self, x):
        values = Tensor(x).tolist()
        if not values:
            return Tensor([0.0] * self.out_features)
        mean = sum(values) / len(values)
        return Tensor([mean] * self.out_features)


class _Flatten(Module):
    def forward(self, x):  # pragma: no cover - trivial flatten
        return Tensor(Tensor(x).tolist())


class _NNNamespace:
    Module = Module
    Sequential = _Sequential
    Conv2d = _Passthrough
    ReLU = _ReLU
    MaxPool2d = _Passthrough
    Flatten = _Flatten
    Linear = _Linear


def _cosine_similarity(x: TensorInput, y: TensorInput, dim: int = 0) -> Tensor:
    x_vals = Tensor(x).tolist()
    y_vals = Tensor(y).tolist()
    if len(x_vals) != len(y_vals):
        raise ValueError("cosine_similarity requires tensors of the same length")
    dot_product = sum(a * b for a, b in zip(x_vals, y_vals))
    x_norm = math.sqrt(sum(a * a for a in x_vals)) or 1.0
    y_norm = math.sqrt(sum(b * b for b in y_vals)) or 1.0
    return Tensor(dot_product / (x_norm * y_norm))


class _Adam:
    def __init__(self, params: Iterable[object], lr: float = 1e-3) -> None:  # pragma: no cover - trivial
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:  # pragma: no cover - trivial
        return None

    def step(self) -> None:  # pragma: no cover - trivial
        return None


class _OptimNamespace:
    Adam = _Adam


class _DataLoader(list):
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial stub
        super().__init__()


class _UtilsNamespace:
    class data:  # pragma: no cover - simple namespace
        DataLoader = _DataLoader


@contextlib.contextmanager
def no_grad():  # pragma: no cover - trivial context manager
    yield


nn = _NNNamespace()
optim = _OptimNamespace()
utils = _UtilsNamespace()
float32 = "float32"

_nn_functional = types.ModuleType("torch.nn.functional")
_nn_functional.cosine_similarity = _cosine_similarity
nn.functional = _nn_functional  # type: ignore[attr-defined]

_utils_module = types.ModuleType("torch.utils")
_utils_data_module = types.ModuleType("torch.utils.data")
_utils_data_module.DataLoader = _DataLoader
_utils_module.data = _utils_data_module
sys.modules.setdefault("torch.nn", nn)  # pragma: no cover - module registration
sys.modules["torch.nn.functional"] = _nn_functional
sys.modules["torch.utils"] = _utils_module
sys.modules["torch.utils.data"] = _utils_data_module


__all__ = [
    "Tensor",
    "tensor",
    "dot",
    "equal",
    "nn",
    "optim",
    "utils",
    "no_grad",
    "Module",
    "float32",
]
