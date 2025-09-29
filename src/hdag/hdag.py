"""HDAG module storing tensor nodes with persistent JSON backing."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import torch

NodeName = str
Edge = Tuple[NodeName, NodeName, float]


@dataclass
class HDAG:
    """Hierarchical Directed Acyclic Graph composed of tensor nodes.

    Nodes and edges are persisted to a JSON file (``storage_path``). The graph can
    be reloaded by instantiating :class:`HDAG` with the same storage path.
    """

    storage_path: Path | str | None = None
    similarity_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None
    nodes: Dict[NodeName, torch.Tensor] = field(init=False, default_factory=dict)
    edges: List[Edge] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path)
        if self.similarity_fn is None:
            self.similarity_fn = self._cosine_similarity
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    def _load(self) -> None:
        if not isinstance(self.storage_path, Path) or not self.storage_path.exists():
            self.nodes = {}
            self.edges = []
            return

        data = json.loads(self.storage_path.read_text())
        nodes_serialised = data.get("nodes", {})
        self.nodes = {name: torch.tensor(values) for name, values in nodes_serialised.items()}
        self.edges = [tuple(edge) for edge in data.get("edges", [])]  # type: ignore[list-item]

    def _save(self) -> None:
        if not isinstance(self.storage_path, Path):
            return
        serialised = {
            "nodes": {name: self._tensor_to_list(tensor) for name, tensor in self.nodes.items()},
            "edges": list(self.edges),
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(serialised, sort_keys=True))

    # ------------------------------------------------------------------
    def add_node(self, name: NodeName, vector: torch.Tensor) -> None:
        """Add or update a node in the DAG."""

        self.nodes[name] = self._as_tensor(vector)
        self._save()

    def add_edge(self, u: NodeName, v: NodeName, weight: float) -> None:
        """Add a directed, weighted edge between nodes ``u`` and ``v``."""

        if u not in self.nodes or v not in self.nodes:
            raise KeyError("Both nodes must exist before adding an edge.")
        self.edges.append((u, v, float(weight)))
        self._save()

    def neighbors(self, node: NodeName) -> List[Tuple[NodeName, float]]:
        """Return outgoing neighbours from ``node``."""

        if node not in self.nodes:
            raise KeyError(f"Node '{node}' does not exist")
        return [(dst, w) for src, dst, w in self.edges if src == node]

    def resonance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute the resonance score (cosine similarity by default)."""

        return self.similarity_fn(x, y)  # type: ignore[return-value]

    @staticmethod
    def _cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
        x_vals = HDAG._tensor_to_list(x)
        y_vals = HDAG._tensor_to_list(y)
        if len(x_vals) != len(y_vals):
            raise ValueError("Vectors must share dimensionality")
        x_norm = math.sqrt(sum(value * value for value in x_vals))
        y_norm = math.sqrt(sum(value * value for value in y_vals))
        if x_norm == 0.0 or y_norm == 0.0:
            raise ValueError("Cannot compute cosine similarity for zero vectors")
        dot_product = sum(a * b for a, b in zip(x_vals, y_vals))
        return dot_product / (x_norm * y_norm)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, item: object) -> bool:
        return isinstance(item, str) and item in self.nodes

    def items(self) -> Iterable[Tuple[NodeName, torch.Tensor]]:
        return self.nodes.items()

    @staticmethod
    def _tensor_to_list(tensor: torch.Tensor) -> List[float]:
        if hasattr(tensor, "tolist"):
            return list(tensor.tolist())
        if hasattr(tensor, "__iter__"):
            return [float(v) for v in tensor]
        return [float(tensor)]

    @staticmethod
    def _as_tensor(vector: torch.Tensor) -> torch.Tensor:
        if isinstance(vector, torch.Tensor):
            return torch.tensor(vector)
        raise TypeError("vector must be a torch.Tensor")
