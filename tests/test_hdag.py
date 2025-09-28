import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from src.hdag.hdag import HDAG


def test_add_node():
    graph = HDAG()
    tensor = torch.tensor([1.0, 2.0])
    graph.add_node("a", tensor)
    assert "a" in graph.nodes
    assert torch.equal(graph.nodes["a"], tensor)


def test_add_edge():
    graph = HDAG()
    graph.add_node("a", torch.tensor([1.0, 0.0]))
    graph.add_node("b", torch.tensor([0.0, 1.0]))

    graph.add_edge("a", "b", 0.5)

    assert ("a", "b", 0.5) in graph.edges


def test_resonance():
    graph = HDAG()
    x = torch.tensor([1.0, 0.0])
    y = torch.tensor([0.0, 1.0])
    z = torch.tensor([2.0, 0.0])

    orthogonal = graph.resonance(x, y)
    parallel = graph.resonance(x, z)

    assert math.isclose(orthogonal, 0.0, abs_tol=1e-6)
    assert math.isclose(parallel, 1.0, abs_tol=1e-6)


def test_neighbors():
    graph = HDAG()
    graph.add_node("a", torch.tensor([1.0, 0.0]))
    graph.add_node("b", torch.tensor([0.0, 1.0]))
    graph.add_node("c", torch.tensor([1.0, 1.0]))

    graph.add_edge("a", "b", 0.3)
    graph.add_edge("a", "c", 0.7)
    graph.add_edge("b", "c", 0.9)

    neighbors = graph.neighbors("a")

    assert ("b", 0.3) in neighbors
    assert ("c", 0.7) in neighbors
    assert len(neighbors) == 2
