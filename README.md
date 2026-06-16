# ai-newsroom-humanizer

**Test-Korpus + Humanizer-Werkstatt für die Frage: Lässt sich Pangram austricksen?**

Aktiver Forschungsprototyp im AI-Newsroom-Stack. Misst empirisch, welche
KI-Humanizer-Strategien Pangrams AI-Detektor unterlaufen — und welche nicht.

Status: **Phase 2 abgeschlossen (5/24 = 21 % Bypass + 24/24 inhalts-treu auf Casdorff).**
Generisches CLI `humanize` produktionsreif. Phase 3 (echtes RL-Training) als Roadmap.

---

## CLI — `humanize` (generisch, jeder Text)

```bash
# Setup einmalig
python3 -m venv .venv && .venv/bin/pip install -e ".[cli]"
# rsync ruediger:Projects/ai-newsroom-humanizer/data/phase2/proxy_* data/phase2/

# Datei -> Datei
humanize artikel.txt -o human.txt

# stdin -> stdout
cat artikel.txt | humanize -

# voller JSON-Trace (Iterations-Historie)
humanize artikel.txt --json -o trace.json

# mit echter Pangram-API-Eval am Ende (~$0.10)
humanize artikel.txt --eval -o human.txt

# Tuning
humanize artikel.txt --threshold 0.25 --max-iters 3 --variants 6
```

**Wie es funktioniert:** Sonnet 4.5 generiert `--variants` parallele Umschreibungen
mit Temperatur-Sweep 0.6–1.1. Der Phase-2-Proxy (BGE-M3 + 2-Layer-MLP, val_MAE 0.29)
bewertet alle Kandidaten lokal auf MPS — die beste wird zur nächsten Iteration. Stopp
bei `proxy_score < threshold` oder `max_iters`. Inhaltstreue wird mit `--eval` per
BGE-Sim ≥ 0.85 verifiziert.

**Kosten** (Phase-2-Setting, 10 Varianten × 5 Iter): ~$1 pro Artikel Sonnet, $0 für
Proxy-Reward. Mit `--eval`: +$0.10 für 2 Pangram-Calls.

---

## Worum es geht

Aufgabe (Auftrag 2026-06-15):
> „Wir wollen zeigen, dass die als KI klassifizierten Artikel mit Hilfe von KI auch wieder in
> menschlich aussehende umgewandelt werden können. Der Inhalt darf aber nicht verfälscht werden."

Test-Korpus: 34 Tagesspiegel-Kommentare von **Stephan-Andreas Casdorff** (Mitherausgeber),
alle von Pangram mit `fraction_ai = 1.0` als KI eingestuft.

Strategie: drei Humanizer-Familien bauen, identische Stichprobe, Pangram pre/post.

---

## Empirische Befunde (Stand 2026-06-15)

| Strategie | Methode | Erfolg (P(AI) < 0.2) | Faithfulness | Kosten/Artikel |
|---|---|---|---|---|
| **02 — Iterative Single-Model** | Sonnet-Loop, Best-of-3, max 5 Iter | **1/25 = 4 %** | 44 % FAITHFUL | $0.35 |
| **03 — Multi-Model Recursive** | Sonnet → GPT-4o → Gemini → Sonnet | **0/5 = 0 %** | n/a (Pilot) | $0.05 |
| **A — Stilometric Prompt** | Casdorff-Stilprofil (28 Human-Artikel) im System-Prompt | **0/5 = 0 %** | (Pilot) | $0.12 |
| **05 — Hybrid Sentence-Mix** | Echte Casdorff-Human-Sätze in den KI-Text injizieren | **2/5 = 40 %** | **DESIGN-BREACH** (semantisch) | $0 (kein LLM) |
| **06 — Mikro-Adversarial** | Whitespace + Unicode-Homoglyph (SpaceInfi, Creo) | als Lift in 05 | n/a | $0 |

**Schluss aus den Daten:** Pangram hält gegen **prompt-engineered Sonnet-Paraphrase** robust
(0–4 %), unabhängig von Iteration, Multi-Model-Mix oder empirisch gemessenem Stil-Profil.
Hybrid-Edit funktioniert (40 %), verletzt aber Inhaltstreue (fremde Casdorff-Sätze aus anderen
Artikeln werden eingestreut). Die wissenschaftlich belegt erfolgreiche Variante — AuthorMist mit
RL-Training gegen Pangram-API — ist als Roadmap dokumentiert (`docs/adr/003-roadmap-authormist.md`).

---

## Verzeichnisstruktur

```
src/humanizer/
  _openrouter.py             — OpenRouter-Client (anthropic/Sonnet, openai/GPT-4o, google/Gemini)
  profile_extractor.py       — Stilometrie aus echten Human-Artikeln (Satzlängen, Vokabular, Anti-Pattern)
  faithfulness.py            — LLM-Judge + deterministische Strukturchecks (n-gram, Namen, Zahlen)
  export.py                  — Pre/Post-Excel + Word-Files mit Pangram-Box
  strategies/
    02_iterative_loop.py     — AuthorMist-light: Sonnet-Loop mit Pangram-Feedback
    03_multimodel.py         — Sonnet → GPT-4o → Gemini → Sonnet recursive
    05_hybrid_edit.py        — Sentence-Mix + Mikro-Adversarial-Layer
    A_stilometric_prompt.py  — Casdorff-Stilprofil im System-Prompt

docs/adr/                    — Architektur-Entscheidungen
data/profiles/casdorff.json  — extrahiertes Casdorff-Stilprofil (28 Artikel, 1117 Sätze)
data/test-corpora/casdorff-2026-06-15/ — alle Pilot-Resultate als JSONL (reproduzierbar)
```

---

## Voraussetzungen

- Python ≥ 3.11 + venv (`uv pip install -e .` oder `pip install httpx anthropic openpyxl python-docx`)
- `~/.config/openrouter/key` — OpenRouter API-Key (für anthropic/openai/google via einem Provider)
- `~/.config/pangram/key` — Pangram API-Key (Account `pangram.com`)
- `~/Projects/ki-check/data/ki-check.db` — Test-Korpus (huGO+-Artikel + Pangram-Werte)

Optional für Roadmap-Phase 2:
- ruediger (M-Class Mac mit MLX) für AuthorMist-RL-Training
- Qwen2.5-3B-Instruct (HuggingFace) als Base-Model

---

## Lauf

```bash
# 1. Stilprofil eines Autors extrahieren (28 Human-Artikel → Satzlängen, Vokabular, Anti-Pattern)
python -m humanizer.profile_extractor --autor "Casdorff" --out data/profiles/casdorff.json

# 2. Humanize-Strategie laufen lassen (gegen 5 AI-eingestufte Artikel als Pilot)
HUMANIZE_LIMIT=5 python -m humanizer.strategies.A_stilometric_prompt

# 3. Faithfulness-Check
python -m humanizer.faithfulness

# 4. Excel + Word-Export
python -m humanizer.export
```

Output landet in `~/Downloads/ai-newsroom-humanizer/<autor>-<datum>/`.

---

## Wissenschaftlicher Stand 2026

Stand der Forschung (Recherche dokumentiert in `docs/adr/001-state-of-research.md`):

- **AuthorMist** (David & Gervais, ETH Zürich 2025, [arXiv:2503.08716](https://arxiv.org/abs/2503.08716))
  Qwen2.5-3B + GRPO mit Detektor-API als Reward → 78,6–96,2 % Bypass auf GPTZero, WinstonAI,
  Originality.ai, Sapling. Inhaltstreue Similarity > 0,94. **Pangram nicht im Eval** — Forschungslücke.
- **AuthorMix** (Saarland 2026-03, [arXiv:2603.23069](https://arxiv.org/abs/2603.23069))
  Layer-wise Adapter Mixing, „handful" Beispiele für neuen Autor. Schlägt GPT-5.1 für low-resource.
  Code noch nicht öffentlich. MLX-tauglich.
- **DAMAGE** (Pangram Labs 2025, [arXiv:2501.03437](https://arxiv.org/abs/2501.03437))
  Pangram greift sich selbst mit fine-tuned Attack-Model an — bleibt nach Selbstangabe robust durch
  „cross-humanizer generalization". Hersteller-Studie, Bias-Vorbehalt.
- **„Base Models Look Human"** (2026, [arXiv:2605.19516](https://arxiv.org/abs/2605.19516))
  Manche Base-LLMs werden bereits ohne Fine-Tuning als Human eingestuft — verschiebt Spiel von
  „austricksen" zu „richtiges Model wählen".

---

## Lizenz / Ethik

Dual-Use-Forschung. Veröffentlichung im AI-Newsroom-Kontext: Nachweis der Robustheits-Grenzen
kommerzieller AI-Detektoren, **NICHT** als Anleitung zur Täuschung in journalistischen oder
akademischen Produktivkontexten. Tests laufen ausschließlich auf eigenem Korpus + eigenen
Tagesspiegel-Artikeln (HMG-Lizenz).

Pangram-API-Calls aus `~/.config/pangram/key` mit Budget-Guard.

---

## Autor

Almagenic / HMG AI-Newsroom · Forschungsprototyp · 2026-06-15
