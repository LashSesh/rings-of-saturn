import math

import pytest

from spiral import Spiral


@pytest.fixture(scope="module")
def _matplotlib():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")
    return pyplot


def test_spiral_point_has_positive_length():
    spiral = Spiral()
    point = spiral.spiral(math.pi / 4)
    length = math.sqrt(sum(coord ** 2 for coord in point))
    assert length > 0


def test_resonance_behaviour():
    spiral = Spiral()
    point = spiral.spiral(math.pi / 3)
    assert pytest.approx(1.0) == spiral.resonance(point, point)

    orthogonal_x = (1, 0, 0, 0, 0)
    orthogonal_y = (0, 1, 0, 0, 0)
    assert abs(spiral.resonance(orthogonal_x, orthogonal_y)) < 1e-6


def test_plot_3d_runs_without_errors(_matplotlib):
    spiral = Spiral()
    fig, _ = spiral.plot_3d(n_points=50)
    _matplotlib.close(fig)
