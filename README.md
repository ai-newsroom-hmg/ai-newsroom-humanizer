# Palimpsest

> Παλίμψηστος (palímpsēstos): wieder abgeschabt. In der Antike: Manuskripte, deren ursprüngliche
> Schrift weggekratzt wurde, um neuen Text aufzunehmen — die alte Schicht aber schimmerte durch.

**Palimpsest** ist ein Forschungs-Werkzeug aus dem HMG AI-Newsroom. Es überschreibt von
KI generierten Text so lange, bis der KI-Detektor [Pangram](https://www.pangram.com/) keinen
Maschinenursprung mehr erkennt — der Inhalt aber semantisch erhalten bleibt.

Anlass war die Affäre um Stephan-Andreas Casdorff (Tagesspiegel, Juni 2026): Pangram hatte
mehrere seiner Meinungsbeiträge als KI-generiert geflaggt. Die Frage: **Lässt sich der
Detektor verlässlich austricksen — und was sagt das über Sinn und Grenze solcher Detektoren?**

Antwort: Ja, mit Best-of-N Mistral-3.2 + BGE-Faithfulness-Gate + Pangram-Ranking erreichen
wir **75 % Doc-Bypass auf einem Casdorff-Härtefall-Korpus (n=12) und 75 % auf 4 OOD-Docs**
(ADR 009, Phase 3b). Auf einem 8.500-Zeichen Meinungsbeitrag: **Pangram-Pre 0.654 → Post 0.000**
(klassifiziert als „Human"). Das Tool ist als wissenschaftlicher Nachweis konzipiert,
nicht als Produktiv-Tool für Täuschung — siehe `ETHICS.md` und Dual-Use-Disclaimer unten.

[![CI](https://github.com/ai-newsroom-hmg/ai-newsroom-humanizer/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-newsroom-hmg/ai-newsroom-humanizer/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Empirische Befunde (Phase 3b, 2026-06-19)

| Sample | Setup | Doc-Bypass | Faith-Rate |
|---|---|---:|---:|
| 12 Casdorff-Härtefälle (Pangram-Pre 1.0) | Best-of-50, BGE ≥ 0.85 | **9/12 = 75 %** | 81 % |
| 4 OOD (verschiedene Autoren) | Best-of-30 | **3/4 = 75 %** | 65 % |
| 8.516-char Meinungsbeitrag | chunked Best-of-24 | **Pangram 0.654 → 0.000** | global BGE 0.846 |

Per-Variant-Bypass-Rate (faithful + Pangram <0.2): **5.7 %**, sehr nahe an Phase-3a-Rate (4.5 %).
Modelliert: Best-of-100 erreicht 99 % Doc-Coverage.

Volle ADR: [`docs/adr/009-phase3b-best-of-50-production.md`](docs/adr/009-phase3b-best-of-50-production.md)

---

## Installation

```bash
# direkt aus GitHub:
pip install git+https://github.com/ai-newsroom-hmg/ai-newsroom-humanizer.git

# oder lokaler Clone für Entwicklung:
git clone https://github.com/ai-newsroom-hmg/ai-newsroom-humanizer.git
cd ai-newsroom-humanizer
pip install -e ".[dev]"
```

**Voraussetzungen:**
- Python ≥ 3.11
- Apple Silicon (MPS) oder CUDA empfohlen für BGE-M3-Embeddings (CPU funktioniert, langsamer)
- API-Keys in `~/.config/<service>/key`:
  - `~/.config/openrouter/key` — OpenRouter für Mistral-3.2-24b-instruct
  - `~/.config/pangram/key` — Pangram Detector-API (für `--eval` + Live-Mode)

---

## Quickstart

```bash
# whole-text Best-of-24 mit Pre/Post-Pangram-Eval (~$1-2)
palimpsest artikel.txt --eval

# long-form chunked Best-of-24 (auto bei >4000 chars)
palimpsest lange-meinung.txt --chunked --eval -o human.txt

# stdin → stdout
cat artikel.txt | palimpsest - --variants 12

# Voller JSON-Trace mit allen Varianten
palimpsest artikel.txt --json -o trace.json --eval

# Rollback-Pfad (Sonnet-Loop, Phase 2, RTO < 30s)
palimpsest artikel.txt --legacy
```

### Dev/Staging-Modi

**Dev** (Default): Live OpenRouter + Live Pangram-API. Echte Performance, echte Kosten.

```bash
palimpsest artikel.txt --env dev --eval   # = default
```

**Staging**: Pangram-Antworten aus Cache (kein API-Cost, deterministisch).
UNBEDINGT-Skill 2 Live-Parity erfüllt: Cache stammt aus echten Live-Calls.

```bash
# 1) Pangram-Cache + Test-Korpus liegen unter tests/staging_corpus/
PALIMPSEST_ENV=staging \
PALIMPSEST_PANGRAM_CACHE=tests/staging_corpus/mini_pangram_cache.json \
palimpsest tests/staging_corpus/sample.txt --eval

# 2) Oder via --env Flag (wins over env-var)
palimpsest tests/staging_corpus/sample.txt --env staging --eval
```

**Optional Ollama** (für Compute-Spar, z.B. lokales `mistral-small-3.2:24b` via Ollama):

```bash
PALIMPSEST_OLLAMA_URL=http://ruediger:11434 palimpsest artikel.txt --env staging
```

*(Ollama-Routing ist als Feature-Flag vorbereitet; ORClient-Anbindung erfolgt in v0.3.)*

---

## Wie es funktioniert (Pipeline)

```
Input-Text
   │
   ├─[Pangram-Pre]──► fraction_ai < 0.2? → Skip
   │
   ├─[Auto-Chunked bei >4000 chars]
   │   └─ split_paragraphs (robust gegen single-\n iCloud-Drift)
   │
   ├─[Best-of-N Mistral-3.2-24b-instruct] (parallel, Temp-Sweep 0.85–1.15)
   │
   ├─[BGE-M3 Multi-Chunk Similarity] (min-Aggregation, MPS/CPU)
   │   └─ Filter ≥ BGE_THRESHOLD (0.85) → faithful-Variants
   │
   ├─[Pangram Live-Rank] (auf faithful Variants)
   │
   └─[Best = (lowest pangram_fraction_ai, highest bge_sim)]
       │
       └─► Output: humanisierter Text
```

Architektur-Details: [`docs/adr/009-phase3b-best-of-50-production.md`](docs/adr/009-phase3b-best-of-50-production.md)

---

## Kosten

| Setup | Per-Artikel-Cost | Bypass-Rate |
|---|---|---|
| Best-of-12 (short Text) | ~$0.50 | ~50 % |
| Best-of-24 (Standard) | ~$1.50 | ~67 % (Phase 3a empirisch) |
| Best-of-50 (Härtefälle) | ~$3.00 | ~75 % (Phase 3b empirisch) |
| Best-of-100 (Maximum) | ~$6.00 | ~99 % (modelliert) |
| Chunked Best-of-24 (lange Texte) | ~$5.00 | Pangram 0 möglich |

Pangram dominiert die Cost (~95 % des Budgets). BGE-Filter ≥ 0.85 vor Pangram spart ~30 %.

---

## Entwicklung

```bash
pip install -e ".[dev]"
pytest tests/ -v        # 16 Tests, läuft ohne API-Keys (staging-mode)
ruff check src/ tests/  # Lint
```

CI: `.github/workflows/ci.yml` — Python 3.11+3.12, pytest + ruff in staging-mode.

---

## Lehren aus Phase 1-3b (siehe ADRs)

1. **Multi-Chunk-BGE-Sim ist Pflicht für lange Texte.** Truncated single-shot BGE
   maskiert Mid-Document-Token-Drift (Pangram 0.0 + BGE 0.94 sah grün aus, Output
   war kaputt). Multi-Chunk min-Aggregation ist konservativ + robust.

2. **iCloud-Sync nivelliert `\n\n` auf `\n`.** Paragraph-Splitter muss Fallback haben.

3. **n=3 ist nie genug für Bypass-Statistiken.** Phase-3-Smoke (n=3) zeigte 100 %, Phase-3a (n=12)
   korrigierte auf 42 %. Minimum: n=12 length-stratifiziert.

4. **Length-Robustheit ist modell-spezifisch.** Pangrams „99 % accuracy on long text" gilt für
   GPT-Class-Paraphrase, nicht für Mistral-3.2. Bei jeder neuen Base-LLM-Familie Length-Strat neu testen.

5. **Best-of-N statt Single-Shot.** Pro-Variant-Bypass-Rate stabil 4.5–5.7 %, Doc-Coverage skaliert
   mit N: `1 − (1−p)^N`.

6. **Faithfulness-Gate vor Bypass-Ranking.** 73 % der Bypass-Varianten wären ohne BGE-Filter
   semantisch verfälscht durchgegangen.

---

## Dual-Use & Ethik

Dieses Tool dokumentiert die **Robustheits-Grenzen kommerzieller AI-Detektoren** im
journalistischen Kontext. Es ist **nicht** als Produktiv-Werkzeug zur Verschleierung
unmarkierter KI-Texte in journalistischen oder akademischen Veröffentlichungen gedacht.

**Berechtigte Nutzung:**
- Wissenschaftliche Untersuchung von Detektor-Robustheit (Adversarial Testing)
- Newsroom-Audits: Wie verlässlich sind Detektoren, die wir intern oder extern einsetzen?
- Forschung zu KI-Generierungs-/Detektions-Wettrüsten (Stand der Forschung: AuthorMist
  arXiv:2503.08716, DAMAGE arXiv:2501.03437, Jabarian/Imas 2025)

**Nicht-berechtigte Nutzung:**
- Verschleierung von KI-Beteiligung in Texten, die unter eigenem Namen veröffentlicht werden
- Umgehung von Kennzeichnungspflichten (EU AI Act Art. 50, Newsroom-internen Richtlinien)

Siehe auch:
- C2PA-Newsroom-Disclosure-Pattern (Memory: `c2pa-newsroom-disclosure-pattern`)
- EU AI Act Art. 50 (Memory: `eu-ai-act-art-50-disclosure-pflicht`)

---

## Roadmap

- **v0.3**: Ollama-Routing für Mistral-3.2-24b auf ruediger (kostenfrei lokal)
- **v0.4**: GRPO + LoRA-Training auf Mistral-3.2 für 1-Shot Bypass (AuthorMist-Pfad, ADR 008 §C)
- **v0.5**: Adversarial Paraphrasing mit Detector-Gradient (arXiv:2506.07001, 87.88 % TPR-Drop)
- **v1.0**: C2PA Content-Credentials-Integration — Tool signiert KI-Beteiligung statt sie zu verschleiern

---

## Lizenz

Apache-2.0. Siehe [`LICENSE`](LICENSE).

## Autoren / Beteiligte

- Gunter Nowy ([@gunternowy](https://github.com/gunternowy)) — Almagenic / HMG AI-Newsroom
- Forschungs-Stack 2026: AI-Newsroom, Mindloom (Knowledge Engine), Enzyme (Vault Catalysts)

---

## Zitation

```bibtex
@software{palimpsest_2026,
  author = {Nowy, Gunter},
  title = {Palimpsest: Empirische Robustheits-Audits von KI-Detektoren in deutschen journalistischen Texten},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ai-newsroom-hmg/ai-newsroom-humanizer},
  version = {0.2.0},
}
```
