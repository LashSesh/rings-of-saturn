# Rings of Saturn

Rings of Saturn ist eine Referenzimplementierung für die Spiral–HDAG–Coupling-Architektur. Das Projekt verbindet ein auditierbares Ledger mit einem tensor-basierten HDAG, Temporal Integrity Capsules (TIC) und Zero-Knowledge Machine Learning (ZKML), um vertrauenswürdige ML-Pipelines mit nachweisbarer Datenherkunft zu ermöglichen.

## Installation

1. Repository klonen
   ```bash
   git clone https://github.com/example/rings-of-saturn.git
   cd rings-of-saturn
   ```
2. Virtuelle Umgebung anlegen (optional, empfohlen)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
3. Abhängigkeiten installieren
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

> Hinweis: Für GPU-basierte Workloads kann eine gesonderte PyTorch-Installation erforderlich sein. Details finden sich in der [PyTorch-Dokumentation](https://pytorch.org/).

## Schnelles Beispiel

Das folgende Beispiel zeigt, wie ein Sensorereignis im Ledger persistiert und anschließend ein Resonanz-Score im HDAG berechnet wird.

```python
import torch
from ledger import Ledger
from hdag.hdag import HDAG

ledger = Ledger()
ledger.add_transaction({"sensor": "lumen", "value": 1337})
block = ledger.create_block()
assert ledger.validate_chain()

hdag = HDAG()
hdag.add_node("sensor", torch.tensor([1.0, 0.5, 0.1]))
hdag.add_node("feature", torch.tensor([0.8, 0.55, 0.05]))
hdag.add_edge("sensor", "feature", 0.9)
print("Resonanz:", hdag.resonance(hdag.nodes["sensor"], hdag.nodes["feature"]))
```

Weitere Informationen zur Architektur, den APIs und exemplarischen Workflows finden sich in der [Projekt-Dokumentation](docs/index.md).

## Lizenz

Dieses Projekt steht unter der [Apache-2.0-Lizenz](LICENSE).
