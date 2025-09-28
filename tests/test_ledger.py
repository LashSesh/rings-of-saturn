"""Tests for the Ledger component."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ledger.ledger import Ledger


def test_add_transaction():
    ledger = Ledger()
    transaction = {"from": "Alice", "to": "Bob", "amount": 10}
    ledger.add_transaction(transaction)

    transaction["amount"] = 15  # ensure the stored transaction was copied

    assert ledger.pending_transactions == [
        {"from": "Alice", "to": "Bob", "amount": 10}
    ]


def test_create_block():
    ledger = Ledger()
    ledger.add_transaction("tx1")
    ledger.add_transaction("tx2")
    block = ledger.create_block()

    assert len(ledger.chain) == 2  # genesis block + new block
    assert ledger.pending_transactions == []
    assert block["index"] == 1
    assert block["transactions"] == ["tx1", "tx2"]
    assert block["prev_hash"] == ledger.chain[0]["hash"]
    assert block["hash"] == ledger.hash_block(block)


def test_create_block_without_transactions():
    ledger = Ledger()

    try:
        ledger.create_block()
    except ValueError as exc:  # pragma: no cover - branch is covered by assertion below
        assert str(exc) == "Cannot create a block with no pending transactions"
    else:
        raise AssertionError("Expected ValueError when creating an empty block")


def test_validate_chain():
    ledger = Ledger()
    ledger.add_transaction("tx1")
    ledger.create_block()
    ledger.add_transaction("tx2")
    ledger.create_block()

    assert ledger.validate_chain() is True

    # Tamper with a block
    ledger.chain[2]["transactions"].append("invalid")

    assert ledger.validate_chain() is False
