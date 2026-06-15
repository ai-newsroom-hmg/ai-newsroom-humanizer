# ADR 003 — Roadmap Phase 2: AuthorMist/AuthorMix mit Pangram-API als Reward

**Status:** Geplant (Vollvariante)
**Datum:** 2026-06-15
**Kontext:** Phase 1 hat empirisch belegt, dass Prompt-Engineering ohne Training Pangram nicht
inhaltstreu bricht. Phase 2 implementiert die wissenschaftlich publizierten Bypass-Methoden,
die gegen Pangram noch nicht getestet wurden.

---

## Ziel

Funktionierenden, **inhaltstreuen** Humanizer bauen + wissenschaftlichen Nachweis liefern:
„Pangram ist gegen detektor-aware RL-Training nicht robust" — die Forschungslücke aus
ADR 001 schließen.

## Architektur

### Schicht 1 — AuthorMist-Light (GRPO ohne Training, nur Inferenz-Optimierung)

```
casdorff_ai_text  → Sonnet generiert N Varianten  → Pangram-Score je Variante
                                                          ↓
                                  Best-of-N + Subset-Iteration
                                                          ↓
                              fraction_ai < threshold? → Output
```

Kein RL-Training, aber strukturierte Such-Strategie statt Greedy:

- Genetic Search: Mutation + Crossover über Best-2-Varianten
- Beam Search: Beam-Width 5, alle Iter Beam-Width vergrößern
- Subset-Optimierung: pro Absatz Pangram-Check, problematische Absätze gezielt rewrite

Erwartung: **15–35 %** (mehr als naive 4 % aus Phase 1).
Aufwand: 1–2 Tage Code, ~$15 Pangram-Budget.

### Schicht 2 — Echte AuthorMist (GRPO mit lokalem Modell)

Stack:
- **Base-Model:** Qwen2.5-3B-Instruct (paper-genau)
- **Training-Framework:** MLX-GRPO ([Doriandarko](https://github.com/Doriandarko/MLX-GRPO))
  oder mlx-tune mit GRPO-Algorithmus
- **Reward-Signal:**
  - Primary: `1 − fraction_ai` (Pangram-API)
  - Secondary: semantische Similarity `cos(BGE-M3(original), BGE-M3(generiert)) > 0.94`
  - Tertiary: Fluency-Critic (Sonnet judges fluency 0–1)
- **Hardware:** ruediger (M-Mac mit MLX)
- **Training:** 50 Casdorff-AI-Artikel als Training-Pool, 25 als Held-Out-Eval

Erwartete Roadmap-Schritte:

1. **Setup MLX-GRPO auf ruediger** (1 Tag): Qwen2.5-3B clonen, MLX-GRPO-Repo installieren,
   GPRO-Reward-Hook auf Pangram-API aufgeräumt schreiben.
2. **Reward-Loop testen** (½ Tag): 100 Samples mit dummy-Reward laufen lassen → Pipeline stabil.
3. **Echtes GRPO-Training** (1–2 Tage Compute): 200–500 Reward-Calls insgesamt, Pangram-API
   im Reward = Budget-kritisch. **Caching Pflicht** (Output → Pangram-Score-Cache).
4. **Eval** (½ Tag): Held-Out 25 Artikel, Vergleich mit Phase-1-Baseline + AuthorMist-Paper-Zahlen.

Erwartung lt. Literatur: **60–90 % Bypass** + Similarity > 0.94.
Aufwand: 3–5 Tage, ~$30–60 Pangram-Budget.

### Schicht 3 — AuthorMix (LoRA-Pool)

Stack:
- Per-Autor-LoRA aus journalistic-style-Korpus (27 Autoren vorhanden)
- Layer-wise Adapter-Mixing (1 Gate-Netzwerk pro Layer)
- Code: nach Paper-Re-Implement ([arXiv:2603.23069](https://arxiv.org/abs/2603.23069)) — kein
  offizielles Repo

Erwartung: höhere Inhaltstreue durch Multi-Adapter-Mix, plus Stil-Quality-Lift.
Aufwand: 2–3 Tage Re-Implement + 1 Tag Training.

---

## Risiken

| Risiko | Mitigation |
|---|---|
| **Pangram-API-Kosten explodieren** im RL-Reward-Loop | Reward-Caching nach Text-SHA, max 500 calls/training-step |
| **Qwen2.5-3B ist zu schwach für deutsche politische Kommentare** | Vergleich mit Llama-3.1-8B als Backup-Base |
| **Pangram detektiert das fine-tuned Modell selbst als Tell** | DAMAGE-Paper-Befund — Cross-Eval gegen das nicht-fine-tuned Base-Model |
| **Dual-Use-Ethik** | Forschungs-Bericht nur intern + an Pangram-Team (responsible disclosure), nicht als Bypass-Tool veröffentlichen |

---

## Pre-Conditions vor Phase 2

1. Phase 1 vollständig dokumentiert ✓
2. ADR 001 + 002 ✓
3. Pangram-Account-Budget mit Owner geklärt (sonst RL nicht möglich)
4. ruediger-Zugang + MLX-Version aktualisiert
5. Casdorff-Korpus auf 50 Train + 25 Eval gesplittet (`data/test-corpora/casdorff-train.jsonl` / `casdorff-eval.jsonl`)
6. responsible-disclosure-Plan mit Pangram-Team abgestimmt (Empfehlung: erst hinter geschlossenen Türen)

---

## Erwartetes Outcome

- **Inhaltstreuer Humanizer**, der Pangram in 60+ % der Fälle täuscht (Casdorff-Korpus)
- **Wissenschaftlich publizierbarer Befund**: Pangram-Robustheit-Audit gegen RL-AuthorMist
- **Empfehlung an Pangram**: konkrete Härtungs-Strategie aus den gefundenen Schwachstellen
