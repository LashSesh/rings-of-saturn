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
    """Simple blockchain-style ledger with a deterministic genesis block."""

    chain: List[Block] = field(default_factory=list)
    pending_transactions: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure a genesis block is always present."""
        if not self.chain:
            genesis_block = self._build_block(
                index=0,
                transactions=[],
                prev_hash="0",
                timestamp=time.time(),
            )
            self.chain.append(genesis_block)

    def add_transaction(self, tx: Any) -> None:
        """Add a transaction to the pending queue."""
        self.pending_transactions.append(deepcopy(tx))

    def create_block(self) -> Block:
        """Create a new block from pending transactions."""
        if not self.pending_transactions:
            raise ValueError("Cannot create a block with no pending transactions")

        prev_hash = self.chain[-1]["hash"]
        block = self._build_block(
            index=len(self.chain),
            transactions=list(self.pending_transactions),
            prev_hash=prev_hash,
            timestamp=time.time(),
        )
        self.chain.append(block)
        self.pending_transactions.clear()
        return block

    def hash_block(self, block: Block) -> str:
        """Return the SHA256 hash of a block."""
        block_content = {k: v for k, v in block.items() if k != "hash"}
        block_string = json.dumps(
            block_content,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

    def validate_chain(self) -> bool:
        """Validate the entire chain."""
        if not self.chain:
            return False

        for index, block in enumerate(self.chain):
            if block.get("index") != index:
                return False

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

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the ledger."""
        return {
            "chain": [deepcopy(block) for block in self.chain],
            "pending_transactions": deepcopy(self.pending_transactions),
        }

    def _build_block(
        self,
        *,
        index: int,
        transactions: List[Any],
        prev_hash: str,
        timestamp: float,
    ) -> Block:
        block: Block = {
            "index": index,
            "timestamp": timestamp,
            "transactions": transactions,
            "prev_hash": prev_hash,
            "hash": "",
        }
        block["hash"] = self.hash_block(block)
        return block
