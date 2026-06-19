# Staging-Test-Korpus

Reproduzierbares Mini-Korpus für CI/staging-Tests.

## Files

| File | Status | Inhalt |
|---|---|---|
| `mini_pangram_cache.json` | committed | 120 Pangram-Cache-Einträge (SHA-keys, keine Volltexte) — sicher zu committen |
| `expected_results.json` | committed | Per-Doc-Erwartungen (Pangram-pre/post, BGE-Range) — keine Texte |
| `docs.jsonl` | **gitignored** | Volltexte 3 Test-Docs aus Casdorff-Korpus + 1 OOD. **NICHT committed** wegen Tagesspiegel-Copyright. |

## Lokal regenerieren

`docs.jsonl` braucht volltext-Zugang. Lokal:

```bash
.venv/bin/python -c "
import json, hashlib
from pathlib import Path

PICKS = [
    ('TSP__5d034c5e0101c339052e6b95a3cb320e826d337c', 'bypass_short', 'Bypass-Success short'),
    ('BB48B8A7-C8AD-496E-BD9A-B2098A622FFD', 'bypass_mid', 'Bypass-Success mid'),
    ('TSP__73fde4c6a3d40d7c686ce5175e54e795e007e443', 'hardcase', 'Härtefall'),
]

eval_docs = {json.loads(l)['doc_id']: json.loads(l)
             for l in open('data/phase2-training-pool/eval.jsonl') if l.strip()}

picked = []
for did, label, note in PICKS:
    d = eval_docs.get(did)
    if not d: continue
    picked.append({'doc_id': did, 'label': label, 'note': note,
                   'volltext': d['volltext'], 'autor': d.get('autor'), 'chars': len(d['volltext'])})

Path('tests/staging_corpus/docs.jsonl').write_text(
    '\n'.join(json.dumps(d, ensure_ascii=False) for d in picked))
print(f'Wrote {len(picked)} docs')
"
```

`eval.jsonl` ist HMG-intern (rsync von ruediger). Ohne lokale Volltexte werden
einige Tests via `pytest.skip()` übersprungen — alle anderen Tests laufen.

## CI ohne Volltexte

Die CI nutzt `mini_pangram_cache.json` für Pangram-Mock-Tests + `expected_results.json`
für Soll-Werte. Volltext-roundtrip-Tests werden in CI geskippt — alle anderen
Funktionalität (env-Resolution, Splitter, CLI-Help) wird voll getestet.
