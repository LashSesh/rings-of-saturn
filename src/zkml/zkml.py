"""Mock implementation of zero-knowledge machine learning primitives."""
from __future__ import annotations

from typing import Any, Callable, Tuple


class ZKML:
    """Utility class to perform zero-knowledge machine learning operations.

    The implementation provided here is a lightweight mock that simulates
    zero-knowledge inference by returning a placeholder proof string. This
    makes it easy to plug in a real zk-SNARK or zk-STARK backend later on
    without having to touch the higher level application code.
    """

    #: Placeholder proof string used by the mock implementation.
    PROOF_PLACEHOLDER = "ZK-PROOF"

    def zk_inference(self, model: Callable[[Any], Any], x: Any) -> Tuple[Any, str]:
        """Run the provided model on ``x`` and return the output and proof.

        Parameters
        ----------
        model:
            A callable that represents the machine learning model. The callable
            must accept ``x`` and return an inference result.
        x:
            The input data for the model.

        Returns
        -------
        Tuple[Any, str]
            A tuple containing the model output ``y`` and a placeholder proof
            string.
        """

        y = model(x)
        proof = self.PROOF_PLACEHOLDER
        return y, proof

    def verify_inference(self, proof: str, x: Any, y: Any) -> bool:
        """Verify the proof for the inference result.

        The mock implementation simply checks whether the provided proof
        matches the placeholder string. ``x`` and ``y`` are accepted to mirror
        the interface of a potential real implementation, but they are not
        used in the verification step.

        Parameters
        ----------
        proof:
            The proof string to verify.
        x:
            The input data used for the inference. Currently unused.
        y:
            The inference result. Currently unused.

        Returns
        -------
        bool
            ``True`` if the provided proof matches the placeholder string,
            ``False`` otherwise.
        """

        return proof == self.PROOF_PLACEHOLDER


__all__ = ["ZKML"]
