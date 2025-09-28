# Architekturübersicht

Die Architektur von **Rings of Saturn** basiert auf dem ursprünglichen Projekt-Blueprint und gliedert sich in fünf miteinander verzahnte Subsysteme:

1. **Ledger Layer** – Verantwortlich für unveränderliche Audit-Trails und die kryptographische Absicherung aller Ereignisse.
2. **Hierarchical Directed Acyclic Graph (HDAG)** – Modelliert abhängige Tensorzustände und stellt Topologie-Operationen für die resonante Datenfusion bereit.
3. **Spiral Orchestrator** – Ein Streaming- und Orchestrierungslayer, der den Datenfluss zwischen Ledger, HDAG und Downstream-Verbrauchern steuert.
4. **Temporal Integrity Capsules (TIC)** – Zeitgekapselte Signaturen, die deterministische Reproduktionen von Modellläufen gewährleisten.
5. **Zero-Knowledge Machine Learning (ZKML)** – Ermöglicht den vertraulichen Nachweis von ML-Inferenzen, ohne sensible Daten offenzulegen.

## Datenfluss

1. Ereignisse gelangen über Gateways in den **Spiral Orchestrator**.
2. Spiral persistiert alle eingehenden Ereignisse im **Ledger**, signiert Blöcke und versieht sie mit kryptographischen Nachweisen.
3. Validierte Zustände werden in den **HDAG** übernommen. Resonanz-Operationen prüfen Tensorbeziehungen und liefern kontextuelle Features.
4. **TICs** kapseln den Zustand jeder Inferenzen (Model, Parameter, HDAG-Schnappschuss) und erstellen reproduzierbare Artefakte.
5. **ZKML** baut auf den TIC-Artefakten auf, generiert Zero-Knowledge-Proofs über die Modellinferenz und stellt sie externen Auditor:innen bereit.

## Komponenteninteraktion

- Das Ledger fungiert als Quelle der Wahrheit. Jede HDAG-Aktualisierung muss auf einem bestätigten Block basieren.
- Spiral kümmert sich um Backpressure, Priorisierung und das Triggern von Re-Computations, sobald TICs oder ZKML-Proofs veraltet sind.
- TICs können sowohl vom Ledger (für Audit) als auch von ZKML (für Proof-Generierung) referenziert werden.

## Deployment

Das Blueprint sieht eine containerisierte Bereitstellung vor:

- **Control Plane**: Spiral Orchestrator (FastAPI/gRPC) und Scheduler.
- **Data Plane**: HDAG-Worker mit GPU-Unterstützung und Ledger-Replikate.
- **Proof Plane**: ZKML-Prover/Verifier-Cluster.

Die Referenzimplementierung in diesem Repository deckt die Kernlogik von Ledger und HDAG ab und stellt Stub-Interfaces für Spiral, TIC und ZKML bereit, damit Erweiterungen nach dem Blueprint erfolgen können.
