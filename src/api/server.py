"""FastAPI server exposing the Rings of Saturn computational services."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from torch import nn

from ledger import Ledger
from hdag.hdag import HDAG
from tic import TIC
from zkml import ZKML
from zkml.zk_inference import build_statement, build_witness


class DummyCNN(nn.Module):
    """Minimal neural network computing the mean of positive inputs."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = _to_list(x)
        if not values:
            return torch.tensor(0.0, dtype=torch.float32)
        mean_val = sum(max(v, 0.0) for v in values) / len(values)
        return torch.tensor(mean_val, dtype=torch.float32)


ledger_service = Ledger()
hdag_service = HDAG()
tic_service = TIC()
zkml_service = ZKML()
ml_model = DummyCNN()


class LedgerTransaction(BaseModel):
    """Payload describing a ledger transaction."""

    sensor: str
    value: float | int


class NodeRequest(BaseModel):
    """Payload for registering a HDAG node."""

    name: str
    vector: List[float]


class EdgeRequest(BaseModel):
    """Payload for connecting two HDAG nodes."""

    source: str = Field(alias="u")
    target: str = Field(alias="v")
    weight: float

    class Config:
        allow_population_by_field_name = True


class ResonanceRequest(BaseModel):
    """Payload describing two node names for resonance computation."""

    source: str
    target: str


class VectorsRequest(BaseModel):
    """Collection of vectors used for TIC condensation."""

    vectors: List[List[float]]


class InvariantRequest(BaseModel):
    """Two TIC states for invariant comparison."""

    state_a: List[float]
    state_b: List[float]


class PredictionRequest(BaseModel):
    """Input vector consumed by the demo CNN model."""

    vector: List[float]


class ZKInferRequest(BaseModel):
    """Input tensor forwarded through the ZKML inference pipeline."""

    vector: List[float]


class ZKVerifyRequest(BaseModel):
    """Payload containing a statement and its proof for verification."""

    statement: str
    proof: str


app = FastAPI(title="Rings of Saturn API", version="0.1.0")


def _to_list(values: Any) -> List[float]:
    if hasattr(values, "tolist"):
        return [float(v) for v in values.tolist()]
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    if hasattr(values, "__iter__"):
        return [float(v) for v in values]
    return [float(values)]


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        return float(value)
    except TypeError:  # pragma: no cover - fallback for tensor stubs
        if hasattr(value, "tolist"):
            data = value.tolist()
            if isinstance(data, list) and data:
                return float(data[0])
        raise


def _tensor(values: List[float]) -> Any:
    tensor_fn = getattr(torch, "tensor")
    dtype = getattr(torch, "float32", None)
    try:
        if dtype is not None:
            return tensor_fn(values, dtype=dtype)
    except TypeError:
        pass
    return tensor_fn(values)


def ledger_add_transaction(tx: Dict[str, Any]) -> Dict[str, Any]:
    ledger_service.add_transaction(tx)
    return {"status": "accepted", "pending": len(ledger_service.pending_transactions)}


def ledger_create_block() -> Dict[str, Any]:
    block = ledger_service.create_block()
    return {"status": "created", "block": block}


def ledger_chain() -> Dict[str, Any]:
    return ledger_service.to_dict()


def hdag_add_node(name: str, vector: List[float]) -> Dict[str, Any]:
    hdag_service.add_node(name, _tensor(vector))
    return {"status": "added", "node": name}


def hdag_add_edge(source: str, target: str, weight: float) -> Dict[str, Any]:
    hdag_service.add_edge(source, target, weight)
    return {"status": "added", "edge": {"source": source, "target": target, "weight": weight}}


def hdag_resonance(source: str, target: str) -> Dict[str, Any]:
    if source not in hdag_service.nodes or target not in hdag_service.nodes:
        raise KeyError("Both nodes must exist to compute resonance.")
    score = float(hdag_service.resonance(hdag_service.nodes[source], hdag_service.nodes[target]))
    return {"resonance": score}


def tic_condense(vectors: List[List[float]]) -> Dict[str, Any]:
    tensors = [_tensor(vec) for vec in vectors]
    condensed = tic_service.update(tensors)
    return {"condensed": tic_service._to_flat_list(condensed)}


def tic_invariant(state_a: List[float], state_b: List[float]) -> Dict[str, Any]:
    tensor_a = _tensor(state_a)
    tensor_b = _tensor(state_b)
    return {"invariant": tic_service.invariant(tensor_a, tensor_b)}


def ml_predict(vector: List[float]) -> Dict[str, Any]:
    input_tensor = _tensor(vector)
    prediction = ml_model(input_tensor)
    return {"prediction": _as_float(prediction)}


def ml_train_demo() -> Dict[str, Any]:
    from ml.demo_training import TrainingConfig, train

    try:
        config = TrainingConfig(epochs=1, batch_size=8, lr=1e-3)
        train(config)
        status = {"status": "completed"}
    except RuntimeError as exc:
        status = {"status": "skipped", "detail": str(exc)}
    return status


def zkml_infer(vector: List[float]) -> Dict[str, Any]:
    input_tensor = _tensor(vector)
    witness = build_witness(ml_model, input_tensor)
    prediction, proof = zkml_service.zk_inference(ml_model, input_tensor)
    statement = build_statement(prediction, witness)
    return {
        "prediction": _as_float(prediction),
        "proof": proof,
        "statement": statement,
    }


def zkml_verify(statement: str, proof: str) -> Dict[str, Any]:
    from zkml.zk_proofs import verify_proof

    return {"valid": bool(verify_proof(statement, proof))}


@app.post("/ledger/add_tx")
def api_ledger_add_tx(request: LedgerTransaction) -> Dict[str, Any]:
    return ledger_add_transaction(request.dict())


@app.post("/ledger/create_block")
def api_ledger_create_block() -> Dict[str, Any]:
    return ledger_create_block()


@app.get("/ledger/chain")
def api_ledger_chain() -> Dict[str, Any]:
    return ledger_chain()


@app.post("/hdag/add_node")
def api_hdag_add_node(request: NodeRequest) -> Dict[str, Any]:
    return hdag_add_node(request.name, request.vector)


@app.post("/hdag/add_edge")
def api_hdag_add_edge(request: EdgeRequest) -> Dict[str, Any]:
    try:
        return hdag_add_edge(request.source, request.target, request.weight)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/hdag/resonance")
def api_hdag_resonance(request: ResonanceRequest) -> Dict[str, Any]:
    try:
        return hdag_resonance(request.source, request.target)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/tic/condense")
def api_tic_condense(request: VectorsRequest) -> Dict[str, Any]:
    if not request.vectors:
        raise HTTPException(status_code=400, detail="vectors must not be empty")
    return tic_condense(request.vectors)


@app.post("/tic/invariant")
def api_tic_invariant(request: InvariantRequest) -> Dict[str, Any]:
    return tic_invariant(request.state_a, request.state_b)


@app.post("/ml/predict")
def api_ml_predict(request: PredictionRequest) -> Dict[str, Any]:
    return ml_predict(request.vector)


@app.post("/ml/train_demo")
def api_ml_train_demo() -> Dict[str, Any]:
    return ml_train_demo()


@app.post("/zkml/zk_infer")
def api_zkml_infer(request: ZKInferRequest) -> Dict[str, Any]:
    return zkml_infer(request.vector)


@app.post("/zkml/verify")
def api_zkml_verify(request: ZKVerifyRequest) -> Dict[str, Any]:
    return zkml_verify(request.statement, request.proof)


def reset_state() -> None:
    """Reset global services to a clean state for testing purposes."""

    ledger_service.chain.clear()
    ledger_service.pending_transactions.clear()
    hdag_service.nodes.clear()
    hdag_service.edges.clear()
    tic_service.state = None

