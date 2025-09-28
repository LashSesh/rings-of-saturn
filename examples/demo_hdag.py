"""Demonstration script for the HDAG component."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from src.hdag.hdag import HDAG


def main() -> None:
    graph = HDAG()

    # Add tensor nodes
    graph.add_node("a", torch.tensor([1.0, 0.0]))
    graph.add_node("b", torch.tensor([0.0, 1.0]))
    graph.add_node("c", torch.tensor([1.0, 1.0]))

    # Connect nodes with directed edges
    graph.add_edge("a", "b", 0.5)
    graph.add_edge("a", "c", 0.8)
    graph.add_edge("b", "c", 0.9)

    # Compute resonance between two nodes
    resonance_value = graph.resonance(graph.nodes["a"], graph.nodes["c"])
    print(f"Resonance between 'a' and 'c': {resonance_value:.4f}")

    # Print neighbors of a node
    neighbors = graph.neighbors("a")
    print("Neighbors of 'a':")
    for neighbor_id, weight in neighbors:
        print(f"  -> {neighbor_id} (weight={weight})")


if __name__ == "__main__":
    main()
