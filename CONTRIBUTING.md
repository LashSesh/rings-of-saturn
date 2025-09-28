# Beitragen zu Rings of Saturn

Vielen Dank für dein Interesse an einem Beitrag!

## Pull-Request-Richtlinien

- Erstelle für jede Änderung einen separaten Branch.
- Halte die Pull Requests klein, thematisch fokussiert und beschreibe die Motivation im PR-Text.
- Ergänze Tests oder Dokumentation für neue Funktionen bzw. Bugfixes.
- Halte dich an den bestehenden Code-Style (PEP 8) und führe statische Analysen/Formatter vor dem Commit aus.

## Tests & Continuous Integration

Vor dem Erstellen eines Pull Requests müssen alle Tests lokal erfolgreich durchlaufen:

```bash
pytest
```

Die GitHub Actions CI führt automatisch `pytest` und den Dokumentations-Build mit MkDocs aus. Stelle sicher, dass beide Schritte ohne Fehler abgeschlossen werden, bevor du deinen PR einreichst.
