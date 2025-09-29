"""Tests for the Spiral module."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import math
import pytest
import torch

from spiral import Spiral


@pytest.fixture(scope="module")
def _matplotlib():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")
    return pyplot


def test_map_returns_expected_tensor():
    spiral = Spiral(a=1.0, b=0.5, c=0.2)
    theta = math.pi / 4
    point = spiral.map(theta)

    expected = torch.tensor(
        [
            spiral.a * math.cos(theta),
            spiral.a * math.sin(theta),
            spiral.b * math.cos(2 * theta),
            spiral.b * math.sin(2 * theta),
            spiral.c * theta,
        ]
    )
    assert isinstance(point, torch.Tensor)
    for actual, expected_value in zip(point.tolist(), expected.tolist()):
        assert actual == pytest.approx(expected_value)


def test_resonance_static_method():
    vector = (1.0, 2.0, 3.0)
    assert pytest.approx(1.0) == Spiral.resonance(vector, vector)
    assert pytest.approx(0.0, abs=1e-6) == Spiral.resonance((1, 0, 0), (0, 1, 0))


def test_plot_projects_to_three_dimensions(_matplotlib):
    spiral = Spiral()
    fig, ax = spiral.plot(n_points=10)
    assert ax.name == "3d"
    _matplotlib.close(fig)
