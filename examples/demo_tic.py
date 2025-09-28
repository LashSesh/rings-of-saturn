"""Demonstration of the Temporal Information Condenser."""
from torch import dot, tensor

from src.tic import TIC


def resonance(x, y):
    return dot(x, y)


def main():
    histories = [
        [tensor([1.0, 0.0]), tensor([0.0, 1.0])],
        [tensor([1.0, 1.0]), tensor([2.0, 2.0])],
    ]

    tic = TIC()
    attractor = tic.condense(histories, resonance)

    print("Selected TIC attractor:", attractor)
    print("Exported representation:", tic.to_dict())


if __name__ == "__main__":
    main()
