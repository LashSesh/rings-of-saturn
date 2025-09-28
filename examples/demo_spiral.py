"""Demonstration script for the Spiral component."""
from __future__ import annotations

import math

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

from spiral import Spiral


def main() -> None:
    spiral = Spiral()

    thetas = [0, math.pi / 4, math.pi / 2, math.pi]
    points = [spiral.spiral(theta) for theta in thetas]

    for theta, point in zip(thetas, points):
        print(f"theta={theta:.2f}: {point}")

    if plt is not None:
        fig, _ = spiral.plot_3d(n_points=200)
        fig.savefig("spiral_plot.png")
        plt.close(fig)
    else:
        print("matplotlib is not installed; skipping plot generation.")

    resonance_value = spiral.resonance(points[0], points[1])
    print(f"Resonance between first two points: {resonance_value:.3f}")


if __name__ == "__main__":
    main()
