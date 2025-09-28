from torch import dot, tensor

from src.tic import TIC


def resonance(x, y):
    return dot(x, y)


def test_condense_selects_highest_resonance_attractor():
    histories = [
        [tensor([1.0, 0.0]), tensor([0.0, 1.0])],
        [tensor([1.0, 1.0])],
    ]

    tic = TIC()
    state = tic.condense(histories, resonance)

    assert state is not None
    assert state.flatten()._values == [1.0, 1.0]


def test_get_state_returns_current_state():
    histories = [[tensor([2.0, 0.0])], [tensor([0.0, 3.0])]]
    tic = TIC()
    tic.condense(histories, resonance)

    state = tic.get_state()

    assert state is not None
    assert state.flatten()._values == tic.state.flatten()._values


def test_to_dict_exports_state():
    tic = TIC()
    tic.state = tensor([4.0, 5.0])

    result = tic.to_dict()

    assert result == {"tic": [4.0, 5.0]}
