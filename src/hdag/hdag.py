"""HDAG module for Spiral-HDAG-TIC-ZKML project."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class HDAG:
    """Hierarchical Directed Acyclic Graph composed of tensor nodes."""

    nodes: Dict[str, torch.Tensor] = field(default_factory=dict)
    edges: List[Tuple[str, str, float]] = field(default_factory=list)

    def add_node(self, node_id: str, tensor: torch.Tensor) -> None:
        """Add a tensor node to the graph."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        self.nodes[node_id] = tensor

    def add_edge(self, src: str, dst: str, weight: float) -> None:
        """Add a directed edge between two nodes with the given weight."""
        if src not in self.nodes:
            raise KeyError(f"Source node '{src}' does not exist")
        if dst not in self.nodes:
            raise KeyError(f"Destination node '{dst}' does not exist")
        self.edges.append((src, dst, float(weight)))

    def resonance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate the cosine similarity between two tensors."""
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Both inputs must be torch.Tensor instances")
        x_norm = x.norm()
        y_norm = y.norm()
        if x_norm.item() == 0 or y_norm.item() == 0:
            raise ValueError("Cannot compute cosine similarity for zero vector")
        numerator = torch.dot(x.flatten(), y.flatten()).item()
        denominator = x_norm.item() * y_norm.item()
        return numerator / denominator

    def neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        """Return neighbors of a given node as (neighbor_id, weight)."""
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist")
        return [(dst, weight) for src, dst, weight in self.edges if src == node_id]
