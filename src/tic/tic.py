"""Implementation of the Temporal Information Condenser (TIC)."""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

from torch import Tensor, tensor

VectorLike = Union[Tensor, Sequence[Union[int, float]], int, float]
History = Sequence[VectorLike]
Histories = Sequence[History]
ResonanceFunction = Callable[[Tensor, Tensor], Union[Tensor, float, int]]


def _ensure_tensor(value: VectorLike) -> Tensor:
    """Convert a supported input type into a :class:`~torch.Tensor`."""
    if isinstance(value, Tensor):
        return value
    return tensor(value)


def _to_float(value: Union[Tensor, float, int]) -> float:
    """Convert resonance scores to plain Python floats."""
    if isinstance(value, Tensor):
        flat = value.flatten()
        return float(sum(flat._values))
    return float(value)


class TIC:
    """Temporal Information Condenser.

    The class selects an attractor from historical trajectories that
    maximises the total resonance with all other points.
    """

    def __init__(self) -> None:
        self.state: Optional[Tensor] = None

    def condense(self, histories: Histories, resonance_func: ResonanceFunction) -> Optional[Tensor]:
        """Condense historical trajectories into a single attractor.

        Args:
            histories: A sequence of histories where each history is a
                sequence of vectors (or tensor-like objects).
            resonance_func: A callable ``F(x, y)`` returning the resonance
                between two points.

        Returns:
            The attractor tensor that maximises the total resonance, or
            ``None`` if no histories are provided.
        """

        points: List[Tensor] = []
        for history in histories:
            for vector in history:
                points.append(_ensure_tensor(vector))

        if not points:
            self.state = None
            return None

        best_point: Optional[Tensor] = None
        best_score: Optional[float] = None

        for candidate in points:
            total_resonance = 0.0
            for other in points:
                total_resonance += _to_float(resonance_func(candidate, other))

            if best_score is None or total_resonance > best_score:
                best_score = total_resonance
                best_point = candidate

        self.state = best_point
        return self.state

    def get_state(self) -> Optional[Tensor]:
        """Return the current TIC state."""

        return self.state

    def to_dict(self) -> dict:
        """Export the TIC state as a dictionary."""

        if self.state is None:
            return {"tic": None}
        values = self.state.flatten()._values[:]
        return {"tic": values}
