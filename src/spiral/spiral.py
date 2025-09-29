"""Spiral module implementing a five dimensional helical mapping."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ModuleNotFoundError:  # pragma: no cover - handled in plot method
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    Axes = Figure = object  # type: ignore[assignment]


@dataclass
class Spiral:
    r"""Spiral that maps angles to five-dimensional torch tensors.

    The mapping follows the parametric equation:

    .. math::
        (a \cos \theta, a \sin \theta, b \cos 2\theta, b \sin 2\theta, c \theta)

    Parameters are configurable to allow different scales on each dimension.
    """

    a: float = 1.0
    b: float = 0.5
    c: float = 0.1

    def map(self, theta: float) -> torch.Tensor:
        """Return the spiral projection for ``theta``.

        Args:
            theta: Angle parameter controlling the position on the spiral.

        Returns:
            A five-dimensional tensor representing the spiral point.
        """

        values = (
            self.a * math.cos(theta),
            self.a * math.sin(theta),
            self.b * math.cos(2 * theta),
            self.b * math.sin(2 * theta),
            self.c * theta,
        )
        return torch.tensor(values)

    def plot(self, n_points: int = 200) -> Tuple[Figure, Axes]:
        """Plot the first three dimensions of the spiral.

        Args:
            n_points: Number of points sampled along the spiral. Must be at least 2.

        Returns:
            The matplotlib figure and axis containing the plot.

        Raises:
            RuntimeError: If matplotlib is not installed in the environment.
            ValueError: If ``n_points`` is less than 2.
        """

        if plt is None or matplotlib is None:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for plotting but is not installed.")

        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        thetas = [i * (4 * math.pi) / (n_points - 1) for i in range(n_points)]
        points = [self.map(theta).tolist() for theta in thetas]
        xs, ys, zs = zip(*((p[0], p[1], p[2]) for p in points))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, label="Spiral 5D → 3D projection")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("x₃")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def resonance(x: Sequence[float], y: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors.

        This helper mirrors the behaviour of the historic implementation and is kept
        for backwards compatibility. The ledger and HDAG modules use torch based
        operations instead.
        """

        if len(x) != len(y):
            raise ValueError("Vectors must share dimensionality for resonance computation.")

        dot_product = sum(float(a) * float(b) for a, b in zip(x, y))
        x_norm_sq = sum(float(a) ** 2 for a in x)
        y_norm_sq = sum(float(b) ** 2 for b in y)

        if x_norm_sq == 0.0 or y_norm_sq == 0.0:
            raise ValueError("Cannot compute cosine similarity for zero magnitude vectors.")

        return dot_product / math.sqrt(x_norm_sq * y_norm_sq)
