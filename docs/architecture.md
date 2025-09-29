# Architecture Overview

The Rings of Saturn blueprint is organised into six tightly integrated subsystems.
Each layer contributes a building block for verifiable, reproducible, and privacy-preserving machine learning workflows.

## Modules

1. **Ledger** – Provides the immutable source of truth. Every ingestion event is batched into cryptographically linked blocks that downstream components reference for audit trails.
2. **Hyperdimensional DAG (HDAG)** – Maintains tensor-native state within a directed acyclic graph. Resonance metrics help correlate features and detect drift across the knowledge base.
3. **Spiral Orchestrator** – Acts as the streaming control plane. It ingests events, applies backpressure, and fans out validated payloads to HDAG, TIC, and ML consumers.
4. **Temporal Integrity Capsules (TIC)** – Capture reproducible snapshots of model artefacts, including parameters, input provenance, and HDAG slices. TICs provide the deterministic anchor for audits.
5. **Machine Learning (ML)** – Supplies PyTorch utilities and training scripts that transform TIC material into deployable inference services.
6. **Zero-Knowledge ML (ZKML)** – Generates verifiable proofs about model predictions. The current mock implementation hashes statements and witnesses, demonstrating how future SNARK/STARK integrations can slot in.

## Data Flow Summary

1. External events enter the system through Spiral, which immediately persists them to the Ledger.
2. Confirmed ledger blocks trigger HDAG updates. Tensor operations compute resonance signals and feed feature stores.
3. TICs package model runs with their dependent ledger and HDAG artefacts to enable deterministic replay.
4. ML services consume TIC inputs to execute inference pipelines.
5. ZKML derives statements from ML predictions, commits to the underlying witnesses, and produces proofs that can be shared with auditors without exposing raw data.

## Deployment Considerations

- **Control Plane** – Spiral API, scheduling logic, and monitoring services.
- **Data Plane** – Scalable HDAG workers (GPU-enabled when necessary) and replicated ledger nodes.
- **Proof Plane** – ZKML proving and verification services, ready to be replaced with production-grade circuits.

The reference implementation in this repository focuses on the Ledger, HDAG, and the mocked ZKML components while leaving Spiral, TIC, and full ML orchestration open for future contributors.
