"""Zero-Knowledge Machine Learning utilities."""
from .zkml import ZKML
from .zk_inference import zk_infer, build_statement, build_witness
from .zk_proofs import generate_proof, verify_proof

__all__ = [
    "ZKML",
    "zk_infer",
    "build_statement",
    "build_witness",
    "generate_proof",
    "verify_proof",
]
