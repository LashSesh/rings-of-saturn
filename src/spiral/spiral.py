"""Spiral component for the Rings of Saturn project."""
from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ModuleNotFoundError:  # pragma: no cover - handled in plot method
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    Axes = Figure = Any  # type: ignore[assignment]


class Spiral:
    """Utility class to work with a five-dimensional spiral."""

    def spiral(self, theta: float, a: float = 1.0, b: float = 0.5, c: float = 0.1) -> Tuple[float, ...]:
        """Compute a point on a five-dimensional spiral.

        Args:
            theta: Angle parameter controlling the position on the spiral.
            a: Radius for the first two dimensions.
            b: Radius for the third and fourth dimensions.
            c: Growth rate for the fifth dimension.

        Returns:
            A tuple representing a point on the 5D spiral.
        """

        x1 = a * math.cos(theta)
        x2 = a * math.sin(theta)
        x3 = b * math.cos(2 * theta)
        x4 = b * math.sin(2 * theta)
        x5 = c * theta
        return (x1, x2, x3, x4, x5)

    def resonance(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Compute the cosine similarity between two five-dimensional points.

        Args:
            x: First 5D point.
            y: Second 5D point.

        Returns:
            The cosine similarity between the two points.

        Raises:
            ValueError: If the vectors have different dimensions or zero magnitude.
        """

        if len(x) != len(y):
            raise ValueError("Vectors must have the same dimensionality")

        dot_product = sum(float(a) * float(b) for a, b in zip(x, y))
        x_norm_sq = sum(float(a) ** 2 for a in x)
        y_norm_sq = sum(float(b) ** 2 for b in y)

        if x_norm_sq == 0.0 or y_norm_sq == 0.0:
            raise ValueError("Cannot compute cosine similarity for zero vector")

        denominator = math.sqrt(x_norm_sq) * math.sqrt(y_norm_sq)
        return dot_product / denominator

    def plot_3d(self, n_points: int = 200) -> Tuple[Figure, Axes]:
        """Plot the first three dimensions of the spiral in 3D space.

        Args:
            n_points: Number of points to sample along the spiral.

        Returns:
            A tuple with the created matplotlib figure and axes.

        Raises:
            RuntimeError: If matplotlib is not installed.
        """

        if plt is None or matplotlib is None:
            raise RuntimeError("matplotlib is required for plotting but is not installed.")

        if n_points <= 1:
            raise ValueError("n_points must be greater than 1 to create a plot")

        thetas = [i * (4 * math.pi) / (n_points - 1) for i in range(n_points)]
        points = [self.spiral(theta) for theta in thetas]
        xs, ys, zs = zip(*((p[0], p[1], p[2]) for p in points))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, label="5D Spiral (3D projection)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        fig.tight_layout()
        return fig, ax
