# ADR 002 — Phase-1-Befunde: Prompt-Engineering bricht Pangram nicht

**Status:** Akzeptiert (negativer Befund)
**Datum:** 2026-06-15
**Kontext:** Phase 1 sollte zeigen, ob Pangram **ohne RL-Training** unter Inhaltserhaltung zu umgehen ist.

---

## Test-Setup

- **Korpus:** 34 Stephan-Andreas-Casdorff-Kommentare (Tagesspiegel, Politik-Ressort), alle von Pangram `fraction_ai = 1.0`, Filter `LENGTH(volltext) > 800` → 25 Test-Kandidaten
- **Threshold „Bypass-Erfolg":** Pangram `fraction_ai < 0.20` nach Humanize
- **Modelle:** OpenRouter — `anthropic/claude-sonnet-4.5`, `openai/gpt-4o-2024-11-20`, `google/gemini-2.0-flash-001`
- **Budget-Guard:** Pangram-Kontingent ≤ €50

## Ergebnisse pro Strategie

### Strategie 02 — Iterativer Sonnet-Loop (AuthorMist-light)

- 25 Artikel × max 5 Iter × Best-of-3 Sonnet-Varianten
- Pangram-API gibt Reward-Signal, beste Variante pro Iter gewinnt
- **Ergebnis: 1/25 = 4 % Bypass-Erfolg**, Ø fraction_ai-Drop 0.05
- Kosten: $8.79 Sonnet
- Faithfulness (LLM-Judge): 11/25 FAITHFUL, 8/25 MINOR_DRIFT, 5/25 SIGNIFICANT_DRIFT
- Strukturelle Schwäche: **Zahlen-Erhalt im Schnitt 30 %** — Sonnet entfernt/reformatiert Zahlen

### Strategie 03 — Multi-Model Recursive

- Sonnet → GPT-4o → Gemini-2.0 → Sonnet (final), 5 Artikel Pilot
- **Ergebnis: 0/5 = 0 %**, fraction_ai bleibt bei 1.000 nach allen 4 Pässen
- Kosten: $0.24
- Belegt: Pangram-Detektion ist nicht modell-spezifisch (Sadasivan-Erwartung nicht auf Pangram übertragbar)

### Strategie A — Stilometric Prompt mit Casdorff-Profil

- Profil-Extraktion aus 28 Casdorff-Human-Artikeln (1117 Sätze):
  - Satzlängen Mean 14.4 / Median 12 / P95 31, Kurzsätze 12 %, Fragen-Quote 4.3 %
  - Top-Vokabular, Anti-Pattern (kein „daher", kein „erstens/zweitens", keine 3-Adjektiv-Listen)
- Sonnet mit kompletter Profil-Stütze im System-Prompt + iterativer Pangram-Feedback-Loop
- **Ergebnis: 0/5 = 0 %**
- Kosten: $0.62

### Strategie 05 — Hybrid Sentence-Mix (Source-Confusion)

- 919 echte Casdorff-Human-Sätze aus 28 Artikeln als Pool
- Alle 4 KI-Sätze einen Casdorff-Human-Satz einsetzen
- Optional: Mikro-Adversarial-Layer (06) — SpaceInfi + Unicode-Homoglyph @ 0.02/0.05 Intensität
- **Ergebnis: 2/5 = 40 % Bypass-Erfolg** (P(AI) → 0.000 für 2 Artikel)
- Kosten: $0 (kein LLM)
- **CAVEAT — Design-Breach:** Fremde Casdorff-Sätze (aus anderen Artikeln) werden eingestreut.
  Inhalt ist **nicht** treu zum Original. Verletzt Auftragsbedingung „Inhalt darf nicht verfälscht werden".

---

## Schluss

**Inhalts-treu UND Pangram-täuschend ist mit Sonnet/GPT-4o/Gemini-Prompt-Engineering NICHT erreichbar.**

Pangram's „Hard Negative Mining with Synthetic Mirrors" + cross-humanizer-Augmentation
(Emi & Spero 2024, S.11–12) generalisiert auf alle drei aktuellen Top-LLMs.

**Was funktioniert (mit Caveat):**

- Hybrid Sentence-Mix: 40 % — aber nicht inhalts-treu
- Mikro-Adversarial allein: vernachlässigbar (Pangram-Marketing bestätigt)

**Was nicht funktioniert:**

- Naiver Sonnet-Loop: 4 %
- Multi-Model Recursive: 0 %
- Stilometrie-geführter Sonnet-Loop: 0 %

→ Wenn Inhaltstreue Priorität bleibt: Phase 2 (RL-Training gegen Pangram-API) ist die einzige
realistische Option — siehe ADR 003.

---

## Reproduzierbarkeit

Alle Ergebnisse in `data/test-corpora/casdorff-2026-06-15/`:

- `01_casdorff_loop_results.jsonl` — Strategie 02, 25 Artikel
- `02_faithfulness.jsonl` — LLM-Judge + Strukturchecks, 25 Artikel
- `03_multimodel_results.jsonl` — Strategie 03, 5 Pilot
- `05_hybrid_edit_results.jsonl` — Strategie 05 + 06, 5 Pilot
- `07_stilometric_results.jsonl` — Strategie A, 5 Pilot
