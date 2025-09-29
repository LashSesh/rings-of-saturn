"""Shared service instances exposed to the dashboard and public API."""
from __future__ import annotations

from ledger import Ledger
from hdag.hdag import HDAG
from tic import TIC
from zkml import ZKML

# Instantiate long-lived services so both the dashboard endpoints and the
# existing API surface operate on the same state.  This mirrors the globals that
# previously lived inside ``api.server`` but allows other modules to import them
# without triggering circular dependencies.
ledger_service = Ledger()
hdag_service = HDAG()
tic_service = TIC()
zkml_service = ZKML()

__all__ = [
    "ledger_service",
    "hdag_service",
    "tic_service",
    "zkml_service",
]
