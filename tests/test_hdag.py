"""Tests for the HDAG module."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import torch

from hdag.hdag import HDAG  # noqa: E402


def test_add_node_and_edge_persist(tmp_path):
    storage = tmp_path / "graph.json"
    graph = HDAG(storage_path=storage)
    vector_a = torch.tensor([1.0, 0.0, 0.0])
    vector_b = torch.tensor([0.0, 1.0, 0.0])
    graph.add_node("a", vector_a)
    graph.add_node("b", vector_b)
    graph.add_edge("a", "b", 0.75)

    # Reload graph to ensure persistence worked
    reloaded = HDAG(storage_path=storage)
    assert "a" in reloaded
    for actual, expected_value in zip(reloaded.nodes["a"].tolist(), vector_a.tolist()):
        assert actual == pytest.approx(expected_value)
    assert ("a", "b", 0.75) in reloaded.edges

    data = json.loads(storage.read_text())
    assert "nodes" in data and "edges" in data


def test_resonance_cosine_similarity():
    graph = HDAG()
    x = torch.tensor([1.0, 0.0])
    y = torch.tensor([0.0, 1.0])
    z = torch.tensor([2.0, 0.0])

    assert graph.resonance(x, z) == pytest.approx(1.0)
    assert graph.resonance(x, y) == pytest.approx(0.0, abs=1e-6)


def test_neighbors_returns_weighted_edges():
    graph = HDAG()
    graph.add_node("root", torch.tensor([1.0]))
    graph.add_node("child", torch.tensor([0.5]))
    graph.add_edge("root", "child", 0.25)

    neighbors = graph.neighbors("root")
    assert neighbors == [("child", 0.25)]
