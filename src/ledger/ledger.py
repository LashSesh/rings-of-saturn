"""Ledger implementation for the Rings of Saturn project."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List
import hashlib
import json
import time


Block = Dict[str, Any]


@dataclass
class Ledger:
    """Simple blockchain-style ledger."""

    chain: List[Block] = field(default_factory=list)
    pending_transactions: List[Any] = field(default_factory=list)

    def add_transaction(self, tx: Any) -> None:
        """Add a transaction to the pending queue."""
        self.pending_transactions.append(tx)

    def create_block(self) -> Block:
        """Create a new block using the pending transactions."""
        prev_hash = self.chain[-1]["hash"] if self.chain else "0"
        block: Block = {
            "index": len(self.chain),
            "timestamp": time.time(),
            "transactions": list(self.pending_transactions),
            "prev_hash": prev_hash,
            "hash": "",
        }
        block_hash = self.hash_block(block)
        block["hash"] = block_hash
        self.chain.append(block)
        self.pending_transactions.clear()
        return block

    def hash_block(self, block: Block) -> str:
        """Return the SHA256 hash of a block."""
        block_content = {k: v for k, v in block.items() if k != "hash"}
        block_string = json.dumps(block_content, sort_keys=True, default=str)
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

    def validate_chain(self) -> bool:
        """Validate the entire chain."""
        for index, block in enumerate(self.chain):
            if index == 0:
                if block.get("prev_hash") != "0":
                    return False
            else:
                prev_block = self.chain[index - 1]
                if block.get("prev_hash") != prev_block.get("hash"):
                    return False
            expected_hash = self.hash_block(block)
            if block.get("hash") != expected_hash:
                return False
        return True

    def to_dict(self) -> Dict[str, List[Block]]:
        """Return a serialisable representation of the chain."""
        return {"chain": [deepcopy(block) for block in self.chain]}
