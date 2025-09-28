"""Demonstration script for the mock zero-knowledge ML pipeline."""

from src.zkml import ZKML


def main() -> None:
    zkml = ZKML()

    # Define a dummy linear model for demonstration purposes.
    def model(x):
        return 3 * x + 1

    x = 4
    y, proof = zkml.zk_inference(model, x)

    print(f"Input: {x}")
    print(f"Model output: {y}")
    print(f"Proof: {proof}")

    is_valid = zkml.verify_inference(proof, x, y)
    print(f"Proof valid: {is_valid}")


if __name__ == "__main__":
    main()
