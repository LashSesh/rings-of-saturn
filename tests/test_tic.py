"""Tests for the Time Information Crystals module."""

import torch

from src.tic import TIC


def test_condense_selects_maximum_resonance_vector() -> None:
    """The vector aligned with most others should be chosen."""

    vectors = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([-1.0, 0.0]),
    ]

    attractor = TIC.condense(vectors)

    # The vector pointing towards the mean of the others should win.
    assert attractor.tolist() == vectors[1].tolist()


def test_tensor_product_matches_kronecker_definition() -> None:
    """Tensor product should coincide with PyTorch's Kronecker product."""

    blocks = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.5, -0.5]),
        torch.tensor([3.0]),
    ]

    combined = TIC.tensor_product(blocks)
    expected = torch.tensor([
        blocks[0].tolist()[0] * blocks[1].tolist()[0] * 3.0,
        blocks[0].tolist()[0] * blocks[1].tolist()[1] * 3.0,
        blocks[0].tolist()[1] * blocks[1].tolist()[0] * 3.0,
        blocks[0].tolist()[1] * blocks[1].tolist()[1] * 3.0,
    ])

    assert combined.tolist() == expected.tolist()


def test_invariant_tolerates_small_numerical_noise() -> None:
    """Invariant check should accept values within tolerance bounds."""

    state = torch.tensor([1.0, 2.0, 3.0])
    perturbed = torch.tensor([1.0 + 1e-7, 2.0 - 5e-7, 3.0 + 2e-7])

    assert TIC.invariant(state, perturbed)
    far = torch.tensor([value + 1e-2 for value in state.tolist()])
    assert not TIC.invariant(state, far)

