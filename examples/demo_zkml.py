"""Demonstration script for the zero-knowledge ML pipeline."""
from __future__ import annotations

import torch

from src.zkml import ZKML, build_statement, build_witness, verify_proof


def main() -> None:
    zkml = ZKML()

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = 3.0
            self.bias = 1.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            values = x.flatten().tolist()
            outputs = [self.weight * v + self.bias for v in values]
            return torch.tensor(outputs)

    model = TinyModel()
    x_eval = torch.tensor([0.5])

    prediction, proof = zkml.zk_inference(model, x_eval)

    witness = build_witness(model, x_eval)
    statement = build_statement(prediction, witness)
    is_valid = verify_proof(statement, proof)

    print(f"Input: {x_eval.flatten().tolist()}")
    print(f"Model output: {prediction.flatten().tolist()}")
    print(f"Proof: {proof}")
    print(f"Proof valid: {is_valid}")


if __name__ == "__main__":
    main()
