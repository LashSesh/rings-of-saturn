"""Time Information Crystals (TIC) utilities.

This module implements the core operations required to build and compare
Temporal Information Crystals.  The condensate selection follows the
resonance maximisation rule described in the project specification.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import List

import torch
from torch import Tensor


class TIC:
    """Utility container for TIC operations.

    The class offers static helpers for condensation, invariant crystal
    construction and equality checks.  An instance can keep track of the last
    condensed state via :meth:`update` for convenience when integrating with
    other modules.
    """

    def __init__(self) -> None:
        self.state: Tensor | None = None

    # ---------------------------------------------------------------------
    # Condensation utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_vectors(vectors: Sequence[Tensor | Sequence[Tensor]]) -> List[Tensor]:
        """Flatten nested tensor collections into a simple list.

        Args:
            vectors: A sequence containing tensors or nested sequences of
                tensors.  The helper enables backwards compatibility with the
                original Block 1 interface that supplied histories of vectors.

        Returns:
            A list of tensors extracted from ``vectors`` in traversal order.

        Raises:
            TypeError: If an element is not a tensor or a tensor sequence.
        """

        flat: List[Tensor] = []
        for element in vectors:
            if isinstance(element, Tensor):
                flat.append(element)
            elif isinstance(element, Sequence):
                flat.extend(TIC._flatten_vectors(element))
            else:  # pragma: no cover - defensive programming
                raise TypeError(
                    "TIC condensation expects tensors or sequences of tensors; "
                    f"received {type(element)!r}."
                )
        return flat

    @staticmethod
    def condense(vectors: Sequence[Tensor | Sequence[Tensor]]) -> Tensor:
        """Select the resonance attractor from the provided vectors.

        The attractor ``x*`` is defined as the vector that maximises the sum of
        cosine similarities with every other vector in the collection.

        Args:
            vectors: A sequence of tensors representing candidate attractors.
                Nested sequences are supported for compatibility with the
                Block 1 API.

        Returns:
            The tensor corresponding to the maximum-resonance attractor.

        Raises:
            ValueError: If no tensors are supplied or the tensors are
                incompatible (different shapes, devices or dtypes).
        """

        flattened = TIC._flatten_vectors(vectors)
        if not flattened:
            raise ValueError("Cannot condense an empty collection of vectors.")

        best_index = 0
        best_score: float | None = None

        for idx, candidate in enumerate(flattened):
            total_resonance = 0.0
            for other in flattened:
                total_resonance += TIC._cosine_similarity(candidate, other)

            if best_score is None or total_resonance > best_score:
                best_score = total_resonance
                best_index = idx

        return flattened[best_index]

    def update(self, vectors: Sequence[Tensor | Sequence[Tensor]]) -> Tensor:
        """Condense the vectors and store the resulting state."""

        self.state = self.condense(vectors)
        return self.state

    # ------------------------------------------------------------------
    # Tensor crystal construction
    # ------------------------------------------------------------------
    @staticmethod
    def tensor_product(blocks: Sequence[Tensor]) -> Tensor:
        """Build the invariant crystal state as a tensor product.

        Args:
            blocks: Sequence of tensors ``B_k`` that will be combined using the
                Kronecker (tensor) product.

        Returns:
            Tensor representing the invariant crystal state ``C_TIC``.

        Raises:
            ValueError: If ``blocks`` is empty.
        """

        if not blocks:
            raise ValueError("Cannot compute a tensor product of zero blocks.")

        kron = getattr(torch, "kron", None)
        if callable(kron):
            try:
                result = blocks[0]
                for block in blocks[1:]:
                    result = kron(result, block)
                return result
            except TypeError:
                pass

        result_values = TIC._to_flat_list(blocks[0])
        for block in blocks[1:]:
            block_values = TIC._to_flat_list(block)
            result_values = [a * b for a in result_values for b in block_values]
        return torch.tensor(result_values)

    # ------------------------------------------------------------------
    # Invariance checks
    # ------------------------------------------------------------------
    @staticmethod
    def invariant(state_a: Tensor, state_b: Tensor, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
        """Check whether two TIC states are approximately invariant.

        Args:
            state_a: First tensor state.
            state_b: Second tensor state.
            atol: Absolute tolerance used for comparison.
            rtol: Relative tolerance used for comparison.

        Returns:
            ``True`` if both tensors share the same shape and are approximately
            equal within the provided tolerances; ``False`` otherwise.
        """

        values_a = TIC._to_flat_list(state_a)
        values_b = TIC._to_flat_list(state_b)

        if len(values_a) != len(values_b):
            return False

        for a_val, b_val in zip(values_a, values_b):
            if abs(a_val - b_val) > atol + rtol * abs(b_val):
                return False
        return True

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------
    def get_state(self) -> Tensor | None:
        """Return the most recently stored TIC state, if any."""

        return self.state

    def to_dict(self) -> dict[str, list[float] | None]:
        """Serialise the stored state into a dictionary for inspection."""

        if self.state is None:
            return {"tic": None}
        return {"tic": TIC._to_flat_list(self.state)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_flat_list(tensor: Tensor) -> List[float]:
        """Convert tensors from real PyTorch or the test stub to a flat list."""

        candidate = tensor
        for attr in ("detach", "cpu"):
            candidate = getattr(candidate, attr, lambda: candidate)()

        if hasattr(candidate, "view"):
            try:
                candidate = candidate.view(-1)
            except Exception:  # pragma: no cover - fallback path
                pass
        elif hasattr(candidate, "flatten"):
            candidate = candidate.flatten()

        if hasattr(candidate, "tolist"):
            values = candidate.tolist()

            def _flatten(value):
                if isinstance(value, (list, tuple)):
                    for item in value:
                        yield from _flatten(item)
                else:
                    yield float(value)

            return list(_flatten(values))

        if hasattr(candidate, "_values"):
            return [float(v) for v in candidate._values]

        raise TypeError("Unsupported tensor representation for TIC operations.")

    @staticmethod
    def _cosine_similarity(tensor_a: Tensor, tensor_b: Tensor) -> float:
        """Compute cosine similarity using PyTorch when available."""

        functional = getattr(getattr(torch, "nn", None), "functional", None)
        if functional is not None and hasattr(tensor_a, "unsqueeze") and hasattr(tensor_b, "unsqueeze"):
            cosine_tensor = functional.cosine_similarity(
                tensor_a.unsqueeze(0), tensor_b.unsqueeze(0), dim=-1, eps=1e-12
            )
            if hasattr(cosine_tensor, "item"):
                return float(cosine_tensor.item())

        values_a = TIC._to_flat_list(tensor_a)
        values_b = TIC._to_flat_list(tensor_b)
        if len(values_a) != len(values_b):
            raise ValueError("Vectors must have matching dimensions to compute cosine similarity.")

        dot_product = sum(a * b for a, b in zip(values_a, values_b))
        norm_a = math.sqrt(sum(a * a for a in values_a))
        norm_b = math.sqrt(sum(b * b for b in values_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


__all__ = ["TIC"]

