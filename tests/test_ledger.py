"""Tests for the Ledger module."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import torch

from ledger.ledger import Ledger  # noqa: E402
from spiral import Spiral  # noqa: E402


def test_create_block_embeds_spiral_projection():
    spiral = Spiral(a=2.0, b=1.0, c=0.5)
    ledger = Ledger(spiral=spiral)
    ledger.add_transaction({"from": "Alice", "to": "Bob", "amount": 10})
    block = ledger.create_block()

    assert block["index"] == 0
    assert block["projection"][-1] == pytest.approx(0.0)
    assert len(block["projection"]) == 5
    assert ledger.validate_chain() is True


def test_chain_validation_detects_tampering():
    ledger = Ledger()
    ledger.add_transaction({"tx": 1})
    ledger.create_block()
    ledger.add_transaction({"tx": 2})
    ledger.create_block()

    assert ledger.validate_chain() is True
    ledger.chain[1]["transactions"].append({"tx": "tampered"})
    assert ledger.validate_chain() is False


def test_projection_matches_spiral():
    spiral = Spiral()
    ledger = Ledger(spiral=spiral)
    ledger.create_block()
    ledger.add_transaction({"value": 1})
    block = ledger.create_block()

    expected = spiral.map(1.0)
    projection_values = list(block["projection"])
    for actual, expected_value in zip(projection_values, expected.tolist()):
        assert actual == pytest.approx(expected_value)
