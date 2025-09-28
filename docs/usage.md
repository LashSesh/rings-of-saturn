# Beispiel-Workflows

## 1. Datenaufnahme und Ledger-Persistenz

1. Empfange Rohereignisse über Spiral (z. B. via gRPC).
2. Transformiere die Payload und füge sie als Transaktion zum `Ledger` hinzu.
3. Erzeuge periodisch einen Block über `create_block()`.
4. Validiere die Chain mit `validate_chain()` und repliziere die Blöcke in den Audit-Speicher.

```python
from ledger import Ledger

ledger = Ledger()
ledger.add_transaction({"sensor": "lumen", "value": 1337})
block = ledger.create_block()
assert ledger.validate_chain()
```

## 2. Tensorfusion im HDAG

1. Lade den letzten Audit-Block und extrahiere die relevanten Tensoren.
2. Aktualisiere den HDAG mittels `add_node()` und `add_edge()`.
3. Nutze `resonance()` zur Berechnung von Feature-Scores.
4. Persistiere die Ergebnisse als Teil einer neuen TIC.

```python
import torch
from hdag.hdag import HDAG

graph = HDAG()
emb = torch.tensor([0.1] * 4)
agg = torch.tensor([0.05, 0.2, 0.15, 0.5])

graph.add_node("embedding", emb)
graph.add_node("aggregate", agg)
graph.add_edge("embedding", "aggregate", weight=0.42)
score = graph.resonance(emb, agg)
```

## 3. TIC- und ZKML-Workflow

1. Verpacke den aktuellen Modell-Run in eine TIC inklusive Ledger- und HDAG-Referenzen.
2. Lasse das TIC-Modul einen Signaturbeweis erstellen und im Ledger registrieren.
3. Übergib die TIC an das ZKML-Modul, um einen Proof für Auditor:innen zu generieren.
4. Verifiziere den Proof und stelle ihn externen Gegenstellen bereit.

```python
from tic import Capsule
from zkml import Proof

capsule = Capsule(model_hash="abc123", hdagsnapshot="snapshot-42")
if capsule.verify():
    proof = Proof.from_capsule(capsule)
    assert proof.verify()
```

> Die Module `tic` und `zkml` sind als Erweiterungspunkte vorgesehen. Dieser Workflow zeigt, wie sie sich nahtlos in die vorhandenen Ledger- und HDAG-Komponenten einfügen.
