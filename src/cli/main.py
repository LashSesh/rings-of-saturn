"""Command line interface orchestrating Rings of Saturn services."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import typer

from api.server import (
    hdag_add_edge,
    hdag_add_node,
    hdag_resonance,
    ledger_add_transaction,
    ledger_chain,
    ledger_create_block,
    ml_predict,
    ml_train_demo,
    tic_condense,
    tic_invariant,
    zkml_infer,
    zkml_verify,
)


app = typer.Typer(help="Interact with the Rings of Saturn computational blocks.")
ledger_cli = typer.Typer(help="Operate the ledger block.")
hdag_cli = typer.Typer(help="Manipulate the HDAG block.")
tic_cli = typer.Typer(help="Analyse TIC states.")
ml_cli = typer.Typer(help="Demonstration machine learning utilities.")
zkml_cli = typer.Typer(help="Zero-knowledge machine learning helpers.")


app.add_typer(ledger_cli, name="ledger")
app.add_typer(hdag_cli, name="hdag")
app.add_typer(tic_cli, name="tic")
app.add_typer(ml_cli, name="ml")
app.add_typer(zkml_cli, name="zkml")


def _echo_json(payload: Any) -> None:
    """Print ``payload`` as a JSON string."""

    typer.echo(json.dumps(payload, sort_keys=True))


def _parse_vector(value: str) -> List[float]:
    """Parse a JSON encoded vector from ``value``."""

    data = json.loads(value)
    if not isinstance(data, list):  # pragma: no cover - defensive
        raise typer.BadParameter("Expected a JSON array of floats.")
    return [float(item) for item in data]


@ledger_cli.command("add-tx")
def cmd_ledger_add_tx(transaction: str) -> None:
    """Add a transaction to the ledger."""

    payload = json.loads(transaction)
    _echo_json(ledger_add_transaction(payload))


@ledger_cli.command("create-block")
def cmd_ledger_create_block() -> None:
    """Create a block from pending transactions."""

    _echo_json(ledger_create_block())


@ledger_cli.command("show")
def cmd_ledger_show() -> None:
    """Display the current ledger chain."""

    _echo_json(ledger_chain())


@hdag_cli.command("add-node")
def cmd_hdag_add_node(name: str, vector: str) -> None:
    """Register ``name`` with ``vector`` in the HDAG."""

    _echo_json(hdag_add_node(name, _parse_vector(vector)))


@hdag_cli.command("add-edge")
def cmd_hdag_add_edge(source: str, target: str, weight: float) -> None:
    """Connect two nodes in the HDAG."""

    _echo_json(hdag_add_edge(source, target, weight))


@hdag_cli.command("resonance")
def cmd_hdag_resonance(source: str, target: str) -> None:
    """Compute the resonance between two nodes."""

    _echo_json(hdag_resonance(source, target))


@tic_cli.command("condense")
def cmd_tic_condense(vectors: str) -> None:
    """Condense vectors into a TIC state."""

    payload = json.loads(vectors)
    _echo_json(tic_condense(payload))


@tic_cli.command("invariant")
def cmd_tic_invariant(state_a: Path, state_b: Path) -> None:
    """Check the invariance of two TIC state files."""

    values_a = json.loads(state_a.read_text())
    values_b = json.loads(state_b.read_text())
    _echo_json(tic_invariant(values_a, values_b))


@ml_cli.command("predict")
def cmd_ml_predict(vector: str) -> None:
    """Forward ``vector`` through the demo CNN."""

    _echo_json(ml_predict(_parse_vector(vector)))


@ml_cli.command("train-demo")
def cmd_ml_train_demo() -> None:
    """Run the demo training routine."""

    _echo_json(ml_train_demo())


@zkml_cli.command("zk-infer")
def cmd_zkml_infer(vector: str) -> None:
    """Execute zero-knowledge inference on ``vector``."""

    _echo_json(zkml_infer(_parse_vector(vector)))


@zkml_cli.command("verify")
def cmd_zkml_verify(statement: str, proof: str) -> None:
    """Verify ``proof`` against ``statement``."""

    _echo_json(zkml_verify(statement, proof))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
