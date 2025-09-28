# API-Referenz

## Ledger

```python
from ledger import Ledger

ledger = Ledger()
ledger.add_transaction({"type": "mint", "amount": 42})
block = ledger.create_block()
assert ledger.validate_chain()
```

### Methoden
- `add_transaction(tx)` – Fügt eine Transaktion zur Warteschlange hinzu.
- `create_block()` – Erstellt einen neuen Block inkl. Hash, leert die Warteschlange und hängt den Block an die Chain an.
- `hash_block(block)` – Berechnet einen SHA256-Hash über alle Blockdaten außer `hash`.
- `validate_chain()` – Validiert die Konsistenz und Prüfsummen der gesamten Chain.
- `to_dict()` – Gibt eine serialisierbare Repräsentation der Chain zurück.

## HDAG

```python
import torch
from hdag.hdag import HDAG

graph = HDAG()
graph.add_node("input", torch.tensor([1.0, 1.0, 1.0]))
graph.add_node("latent", torch.tensor([0.0, 1.0, 2.0]))
graph.add_edge("input", "latent", weight=0.9)
score = graph.resonance(graph.nodes["input"], graph.nodes["latent"])
```

### Methoden
- `add_node(node_id, tensor)` – Registriert einen Tensor-Knoten.
- `add_edge(src, dst, weight)` – Fügt eine gerichtete Kante mit Gewicht hinzu.
- `resonance(x, y)` – Berechnet die Kosinus-Ähnlichkeit zwischen zwei Tensoren.
- `neighbors(node_id)` – Liefert alle ausgehenden Nachbarn.

## Spiral

Das `spiral`-Paket definiert die orchestrierenden Services. In der Referenzimplementierung sind die Interfaces als Platzhalter ausgelegt:

```python
from spiral import EventRouter

router = EventRouter()
router.register_sink("ledger", ledger_handler)
```

Die produktive Ausgestaltung umfasst Streaming-Endpunkte, Priorisierungslogik und Backpressure-Steuerung.

## Temporal Integrity Capsules (TIC)

TICs kapseln Modellzustände mitsamt Hashes und HDAG-Snapshots. Die API sieht Prüfroutinen vor, die vor jeder Auslieferung eines Artefakts laufen:

```python
from tic import Capsule

capsule = Capsule(model_hash, hdagsnapshot)
assert capsule.verify()
```

## Zero-Knowledge Machine Learning (ZKML)

Das `zkml`-Modul integriert Prover/Verifier-Workloads:

```python
from zkml import Proof

proof = Proof.from_capsule(capsule)
assert proof.verify()
```

> **Hinweis:** Spiral, TIC und ZKML sind in der Referenzimplementierung als Stubs angelegt. Die vollständige Funktionalität folgt dem Blueprint und lässt sich auf dieser Basis implementieren.
