"""Custom loss functions used by the ML integration layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import List

import torch
from torch import Tensor, nn

from src.tic import TIC


class ResonanceLoss(nn.Module):
    """Augment a base loss with a resonance alignment term.

    The loss encourages model features ``h_θ(x)`` to align with the TIC
    condensate ``x*`` computed from the feature batch.  A cosine similarity
    penalty keeps the model near the resonance attractor.
    """

    def __init__(self, base_loss: nn.Module, lambda_: float = 0.1) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.lambda_ = lambda_

    @staticmethod
    def _ensure_feature_list(features: Tensor | Sequence[Tensor]) -> List[Tensor]:
        """Convert input features into a flat list of tensors."""

        if isinstance(features, Tensor):
            if getattr(features, "ndim", 1) == 1:
                return [features]
            return [sample for sample in features]

        feature_list = list(features)
        if not feature_list:
            raise ValueError("ResonanceLoss requires at least one feature vector.")
        return feature_list

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        features: Tensor | Sequence[Tensor],
    ) -> Tensor:
        """Compute the resonance-augmented loss.

        Args:
            predictions: Raw model outputs used by the base task loss.
            targets: Ground-truth targets for the base loss.
            features: Feature vectors ``h_θ(x)`` used to compute the TIC
                condensate.  The argument can be either a tensor of shape
                ``(batch, feature_dim)`` or an iterable of tensors.

        Returns:
            Scalar tensor containing the augmented loss value.
        """

        task_loss = self.base_loss(predictions, targets)

        feature_list = self._ensure_feature_list(features)
        condensate = TIC.condense(feature_list)
        condensate_for_similarity = getattr(condensate, "detach", lambda: condensate)()

        functional = getattr(getattr(torch, "nn", None), "functional", None)
        can_use_torch = (
            functional is not None
            and hasattr(feature_list[0], "unsqueeze")
            and hasattr(condensate_for_similarity, "unsqueeze")
            and hasattr(torch, "stack")
        )

        if can_use_torch:
            similarities = [
                functional.cosine_similarity(
                    feature.unsqueeze(0), condensate_for_similarity.unsqueeze(0), dim=-1, eps=1e-12
                ).squeeze(0)
                for feature in feature_list
            ]
            mean_similarity = torch.stack(similarities).mean()
            resonance_penalty = 1.0 - mean_similarity
            return task_loss + self.lambda_ * resonance_penalty

        similarity_values = [
            TIC._cosine_similarity(feature, condensate_for_similarity) for feature in feature_list
        ]
        mean_similarity = sum(similarity_values) / len(similarity_values)
        penalty_value = 1.0 - mean_similarity

        if isinstance(task_loss, Tensor):
            scalar = float(task_loss.item())
            total = scalar + self.lambda_ * penalty_value
            tensor_kwargs: dict[str, object] = {}
            dtype = getattr(task_loss, "dtype", None)
            device = getattr(task_loss, "device", None)
            if dtype is not None:
                tensor_kwargs["dtype"] = dtype
            if device is not None:
                tensor_kwargs["device"] = device
            try:
                return torch.tensor(total, **tensor_kwargs)
            except TypeError:
                return torch.tensor(total)

        return task_loss + self.lambda_ * penalty_value


__all__ = ["ResonanceLoss"]

