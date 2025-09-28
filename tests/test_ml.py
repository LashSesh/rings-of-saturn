import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from src.spiral.loss import ResonanceLoss


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_vals = prediction.flatten()._values
    target_vals = target.flatten()._values
    if len(pred_vals) != len(target_vals):
        raise ValueError("prediction and target must have the same size")
    error = [(p - t) for p, t in zip(pred_vals, target_vals)]
    return torch.tensor(sum(e * e for e in error))


class IdentityModel(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def test_resonance_loss_rewards_resonant_predictions():
    model = IdentityModel()
    tic = torch.tensor([0.0, 1.0])
    target = torch.tensor([0.0, 0.0])
    loss_fn = ResonanceLoss(mse_loss, lambda_weight=0.5)

    non_resonant_input = torch.tensor([1.0, 0.0])
    resonant_input = torch.tensor([0.0, 1.0])

    non_resonant_prediction = model(non_resonant_input)
    resonant_prediction = model(resonant_input)

    non_resonant_loss = loss_fn(non_resonant_prediction, target, tic).item()
    resonant_loss = loss_fn(resonant_prediction, target, tic).item()

    assert math.isclose(resonant_loss, 1.0, rel_tol=1e-6)
    assert math.isclose(non_resonant_loss, 1.5, rel_tol=1e-6)
    assert resonant_loss < non_resonant_loss
