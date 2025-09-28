"""Demonstration of the ResonanceLoss in a toy classification loop."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch

from src.spiral.loss import ResonanceLoss


class LinearClassifier(torch.nn.Module):
    """Very small linear model backed by the lightweight torch shim."""

    def __init__(self) -> None:
        super().__init__()
        self.weights = torch.tensor([0.0, 0.0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = features.flatten()._values
        weights = self.weights.flatten()._values
        score = sum(w * x for w, x in zip(weights, values))
        return torch.tensor(score)

    def update(self, features: torch.Tensor, gradient: float, lr: float) -> None:
        values = features.flatten()._values
        weights = self.weights.flatten()._values
        self.weights = torch.tensor([w - lr * gradient * x for w, x in zip(weights, values)])


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = prediction.item() - target.item()
    return torch.tensor(diff * diff)


def generate_dummy_data() -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    samples = [
        (torch.tensor([1.0, 0.0]), torch.tensor(0.0)),
        (torch.tensor([0.0, 1.0]), torch.tensor(1.0)),
        (torch.tensor([0.8, 0.2]), torch.tensor(0.0)),
        (torch.tensor([0.2, 0.9]), torch.tensor(1.0)),
    ]
    return samples


def main() -> None:
    model = LinearClassifier()
    tic_attractor = torch.tensor([0.0, 1.0])
    loss_fn = ResonanceLoss(mse_loss, lambda_weight=0.2)
    data = list(generate_dummy_data())

    for epoch in range(5):
        total_loss = 0.0
        for features, label in data:
            prediction = model(features)
            loss = loss_fn(prediction, label, tic_attractor)
            total_loss += loss.item()

            # Basic gradient descent step for the MSE component.
            gradient = 2.0 * (prediction.item() - label.item())
            # Encourage the model to align with the TIC by nudging the gradient.
            resonance = loss_fn.resonance(prediction, tic_attractor)
            gradient -= loss_fn.lambda_weight * (1.0 - resonance)
            model.update(features, gradient, lr=0.1)

        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, weights={model.weights.flatten()._values}")

    final_alignment = loss_fn.resonance(model(torch.tensor([0.0, 1.0])), tic_attractor)
    print(f"Final resonance with TIC: {final_alignment:.3f}")


if __name__ == "__main__":
    main()
