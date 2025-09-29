"""Tests for the ResonanceLoss implementation."""

import pytest
import torch
from torch import nn

from src.ml import ResonanceLoss


functional = getattr(getattr(torch, "nn", None), "functional", None)
if functional is None or not hasattr(torch, "stack"):
    pytest.skip("Real PyTorch installation required for ResonanceLoss tests.", allow_module_level=True)


def test_resonance_loss_exceeds_base_loss_when_misaligned() -> None:
    """ResonanceLoss should add a positive penalty when features disagree."""

    logits = torch.tensor([[2.0, -1.0], [0.5, 0.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    base_loss = nn.CrossEntropyLoss()
    criterion = ResonanceLoss(base_loss, lambda_=0.5)

    resonance_value = criterion(logits, targets, features)
    base_value = base_loss(logits, targets)

    assert torch.isclose(
        resonance_value - base_value, torch.tensor(0.25), atol=1e-5
    ), "Penalty should equal lambda * (1 - mean_cosine)"


def test_resonance_loss_backward_pass() -> None:
    """Gradients should flow through the augmented loss."""

    logits = torch.tensor([[0.1, 0.9], [0.2, 0.8]], requires_grad=True)
    targets = torch.tensor([1, 1])
    features = torch.tensor([[0.2, 0.8], [0.8, 0.2]], requires_grad=True)

    criterion = ResonanceLoss(nn.CrossEntropyLoss(), lambda_=0.3)
    loss = criterion(logits, targets, features)
    loss.backward()

    assert logits.grad is not None
    assert features.grad is not None

