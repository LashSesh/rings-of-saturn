"""Demonstration of the Ledger component."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ledger import Ledger


def main() -> None:
    ledger = Ledger()
    ledger.add_transaction({"from": "Alice", "to": "Bob", "amount": 42})
    ledger.add_transaction({"from": "Charlie", "to": "Dana", "amount": 7})
    ledger.create_block()

    ledger.add_transaction({"from": "Eve", "to": "Frank", "amount": 3})
    ledger.create_block()

    for block in ledger.chain:
        print(block)


if __name__ == "__main__":
    main()
