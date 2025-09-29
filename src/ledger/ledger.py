"""Simple ledger implementation with spiral projections for each block."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from spiral import Spiral


Block = Dict[str, Any]


@dataclass
class Ledger:
    """Dictionary-based blockchain storing spiral projections for each block."""

    spiral: Spiral = field(default_factory=Spiral)
    chain: List[Block] = field(default_factory=list)
    pending_transactions: List[Dict[str, Any]] = field(default_factory=list)

    def add_transaction(self, tx: Dict[str, Any]) -> None:
        """Add a transaction to the queue of pending transactions."""

        if not isinstance(tx, dict):
            raise TypeError("Transactions must be dictionaries.")
        self.pending_transactions.append(dict(tx))

    def create_block(self) -> Block:
        """Create a new block from pending transactions and append it to the chain."""

        index = len(self.chain)
        prev_hash = self.chain[-1]["hash"] if self.chain else "0"
        projection = self.spiral.map(float(index))
        block: Block = {
            "index": index,
            "timestamp": time.time(),
            "transactions": [dict(tx) for tx in self.pending_transactions],
            "prev_hash": prev_hash,
            "projection": self._tensor_to_list(projection),
        }
        block_hash = self._hash_block(block)
        block["hash"] = block_hash
        self.chain.append(block)
        self.pending_transactions.clear()
        return block

    def validate_chain(self) -> bool:
        """Validate the hash pointers across the chain."""

        for index, block in enumerate(self.chain):
            expected_prev_hash = "0" if index == 0 else self.chain[index - 1]["hash"]
            if block.get("prev_hash") != expected_prev_hash:
                return False
            if block.get("hash") != self._hash_block(block):
                return False
        return True

    def _hash_block(self, block: Block) -> str:
        """Compute the SHA256 hash of ``block`` excluding the ``hash`` field."""

        block_content = {k: v for k, v in block.items() if k != "hash"}
        block_string = json.dumps(block_content, sort_keys=True, default=self._json_default)
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

    @staticmethod
    def _json_default(value: Any) -> Any:
        """JSON serializer for torch tensors and other objects."""

        if hasattr(value, "tolist"):
            return list(value.tolist())
        raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")

    @staticmethod
    def _tensor_to_list(tensor: Any) -> List[float]:
        if hasattr(tensor, "tolist"):
            return list(tensor.tolist())
        if hasattr(tensor, "__iter__"):
            return [float(v) for v in tensor]
        return [float(tensor)]

    def to_dict(self) -> Dict[str, List[Block]]:
        """Return a JSON serialisable representation of the ledger."""

        return {"chain": [dict(block) for block in self.chain]}
