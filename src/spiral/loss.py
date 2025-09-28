"""Loss functions for Spiral ML integrations."""
from __future__ import annotations

from typing import Callable, Union

import torch

Scalar = Union[float, int]
LossValue = Union[Scalar, torch.Tensor]


def _to_float(value: LossValue) -> float:
    """Convert supported loss outputs to a floating point number."""
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)


class ResonanceLoss(torch.nn.Module):
    """Combine task loss with a resonance-based regularizer.

    Parameters
    ----------
    task_loss:
        Callable returning the base loss for the task. It must accept the
        prediction tensor and target tensor and return either a Python scalar
        or a :class:`torch.Tensor` containing a single value.
    lambda_weight:
        Weight applied to the resonance penalty term. Must be non-negative.
    """

    def __init__(self, task_loss: Callable[[torch.Tensor, torch.Tensor], LossValue], lambda_weight: float = 1.0) -> None:
        super().__init__()
        if lambda_weight < 0:
            raise ValueError("lambda_weight must be non-negative")
        self.task_loss = task_loss
        self.lambda_weight = float(lambda_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, tic: torch.Tensor) -> torch.Tensor:
        """Compute the total loss value.

        The total loss is defined as ``L_task + lambda_weight * (1 - F)``, where
        ``F`` is the cosine similarity between the prediction and the TIC
        attractor.
        """

        base_loss = self.task_loss(prediction, target)
        base_value = _to_float(base_loss)
        resonance = self.resonance(prediction, tic)
        total = base_value + self.lambda_weight * (1.0 - resonance)
        return torch.tensor(total)

    def resonance(self, prediction: torch.Tensor, tic: torch.Tensor) -> float:
        """Return the cosine similarity between prediction and TIC."""

        pred_flat = prediction.flatten()
        tic_flat = tic.flatten()
        pred_norm = pred_flat.norm().item()
        tic_norm = tic_flat.norm().item()
        if pred_norm == 0.0 or tic_norm == 0.0:
            return 0.0
        similarity = torch.dot(pred_flat, tic_flat).item() / (pred_norm * tic_norm)
        # Clamp for numerical stability in edge cases where rounding causes overflow
        return max(-1.0, min(1.0, similarity))


__all__ = ["ResonanceLoss"]
