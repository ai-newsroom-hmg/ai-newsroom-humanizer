# Changelog

Alle nennenswerten Änderungen an Palimpsest. Format: [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
Versionierung: [SemVer](https://semver.org/lang/de/).

## [0.2.0] — 2026-06-19

### Hinzugefügt
- **Palimpsest-Brand**: Tool umbenannt von `humanize` zu `palimpsest`. Legacy-Alias
  `humanize` bleibt als Rollback-Pfad erhalten (ADR 008).
- **Mistral-3.2-24b-instruct Best-of-N**: Neue Standard-Pipeline mit parallelem
  Best-of-N + Temperatur-Sweep statt iterativer Loop (ADR 008 Phase 3a).
- **Multi-Chunk BGE-M3 Similarity** mit min-Aggregation: ersetzt truncated single-shot
  BGE-Sim, fängt Mid-Document-Token-Drift in langen Texten ab (Lesson 2026-06-19).
- **Chunked-Mode** (`--chunked`, auto bei >4000 chars): paragraph-wise Best-of-N mit
  per-Absatz Faithfulness-Gate.
- **Robust Paragraph-Splitter**: Fallback auf single-newline (iCloud-Sync nivelliert
  oft `\n\n` auf `\n`).
- **Dev/Staging-Modi**: `--env dev|staging`. Staging nutzt Pangram-Cache (kein API-Cost),
  Test-Korpus-Subset in `tests/staging_corpus/` für CI-Regression.
- **Pangram-Cache-Mock**: `PangramClient(mock_cache_path=...)` lädt Cache statt API.
  Cache-Miss raised explizit (kein silent live-fallback).
- **GitHub Actions CI**: Python 3.11+3.12 Matrix, pytest + ruff in staging-mode.
- **Pytest-Suite** (16 Tests): env-Resolution, Pangram-Mock, Splitter-Robustheit, CLI-Help.
- **Phase 3b Excel-Export** (`palimpsest.phase3b_export`): 4-Sheet-Workbook +
  Best-Variants pro Doc als TXT.
- **ADR 009**: Phase 3b Production-Pipeline + 75 % Doc-Bypass empirisch.
- **Meta-QA Go/No-Go-Bericht**: `docs/phase3b-go-no-go-2026-06-19.md` (UNBEDINGT-Skill 1).
- **Rollback-Skript**: `scripts/rollback-to-sonnet.sh` (UNBEDINGT-Skill 3) + `--legacy` Flag.

### Geändert
- Default-Model: `anthropic/claude-sonnet-4.5` → `mistralai/mistral-small-3.2-24b-instruct`.
- BGE-Filter passiert **vor** Pangram (Cost-Save ~30 %).
- `max_tokens` dynamisch: `max(3000, len(input) * 0.6)`.
- Pyproject: License Apache-2.0, Classifiers, dev-Extras (pytest + ruff).

### Behoben
- `(value or 1.0) < 0.2` mit `value=0.0` → falsche False (jetzt explizit `is not None`).
- `splitlines()` splittete auf U+2028 in Mistral-Outputs → ersetzt durch `.split("\n")`.
- BGE-M3 MPS-OOM bei langen Texten: batch_size=1 + Multi-Chunk-Encoding.

### Empirische Resultate Phase 3b (Nightly 2026-06-19, 26.5 min, $36)
- Casdorff Best-of-50 (12 Docs): **9/12 = 75 % Doc-Bypass**
- OOD Best-of-30 (4 Docs): **3/4 = 75 %**
- Demo-Target chunked Best-of-24 (8516 chars): **Pangram 0.654 → 0.000 (Human)**

---

## [0.1.0] (v0.1-sonnet-loop tag) — 2026-06-17

### Phase 2 final (ADR 007)
- Sonnet 4.5 iterativer Loop mit Phase-2-Proxy (BGE-M3 + 2-Layer-MLP, val_MAE 0.29).
- 5/24 = 21 % Bypass auf Casdorff-short, 0/19 auf long.
- Generisches `humanize` CLI.

### Phase 3a (ADR 008) — Mistral-Replikation
- Mistral-Small-3.2-24b + Best-of-24 + BGE-Sim ≥ 0.85.
- n=12 length-stratifiziert, 42 % Doc-Bypass auf Casdorff.
- Length-Hypothese aus ADR 007 widerlegt: Mistral nicht Length-robust.

### Phase 2 hardcoded (ADR 006)
- StealthRL zero-shot scheitert auf Deutsch.

### Phase 1 (ADR 001-005)
- State of Research, AuthorMist Roadmap, Hybrid-Vergleich-Eval.
- Empirie: Sonnet-Loop 4 % Bypass, Multi-Model 0 %, Hybrid-Edit 40 % (Faith-Breach).
