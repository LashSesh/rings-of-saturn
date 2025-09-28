"""Tests for the end-to-end Rings of Saturn pipeline."""

from examples.demo_pipeline import process_transaction


def test_process_transaction_creates_tic() -> None:
    tx = {"from": "alice", "to": "bob", "amount": 42}

    tic_state = process_transaction(tx)

    assert isinstance(tic_state, dict)
    assert tic_state["count"] == 1
    assert len(tic_state["points"]) == 1

    point = tic_state["points"][0]
    assert {"radius", "angle", "height"} <= set(point.keys())
    assert point["radius"] > 0

    centroid = tic_state["centroid"]
    assert centroid == point
