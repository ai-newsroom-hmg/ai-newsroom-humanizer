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

---

## Optimized Setup — der realistische $5-15-Pfad auf ruediger

Der "naive" Kostenrahmen ($30-60) aus den vorigen Sections gilt nur für Paper-1:1-Replikation.
Mit drei Hebeln drückt sich das auf **$5-15** und **2-3 h Compute**:

### Hebel 1 — Proxy-Reward (statt direkter Pangram-API)

```
Phase 1: 200 Pangram-Calls einmalig sammeln (Texte + Scores)  -> $4
Phase 2: BGE-M3 Embedding + 2-Layer-MLP-Head trainieren       -> $0 (ruediger MLX)
Phase 3: GRPO nutzt den Proxy als Reward                       -> $0 / Call
Phase 4: Drift-Check alle 100 Steps mit 1 echtem Pangram-Call -> ~20 Calls = $0.40
Phase 5: Held-out-Eval 50-100 echte Pangram-Calls              -> $2
```

**Spart ~90 %** der Pangram-Calls, weil die teure API nur initial + sparse Verification gebraucht wird.

### Hebel 2 — MLX-spezifische Hardware-Optimierungen auf ruediger

| Optimierung | Speedup-Faktor | Implementations-Aufwand |
|---|---|---|
| MLX-GRPO statt PyTorch-CPU | 10× (laut [Doriandarko-Repo](https://github.com/Doriandarko/MLX-GRPO)) | 0 — drop-in |
| 4-bit Quantization (`mlx-community/Qwen2.5-3B-Instruct-4bit`) | 4× weniger Memory, 2× Forward | 0 — Model swap |
| LoRA-only-Training (mlx-lm-lora, `rank=16`, `q_proj+v_proj`) | 5-10× faster Convergence | 0 — default |
| Batch-Size 16-32 (statt 4) | 2-3× Throughput | trivial |
| Frozen Base + Adapter-Only | 3-4× faster pro Step | mlx-tune Config |

### Hebel 3 — Replay-Buffer + Off-Policy-Sampling

Gespeicherte (Text, Reward)-Paare aus früheren Training-Steps wiederverwerten. Klassischer
DQN/PPO-Trick. Reduziert API-Druck zusätzlich um 30-50 %.

### Realistische Lauf-Bilanz auf ruediger

```
Phase 1: Initial Sampling          30 min   200 Pangram-Calls = $4
Phase 2: BGE-M3-Proxy-Train        10 min   $0 (lokal MLX)
Phase 3: GRPO LoRA-Train           1-2 h    $0 (Proxy als Reward)
Phase 4: Drift-Verification        5 min    20 echte Pangram-Calls = $0.40
Phase 5: Held-out-Eval             20 min   100 Calls = $2
                                  ─────    ────────
                                  ~2-3 h    $6-8 total (Worst-Case-Limit $15)
```

→ **20× billiger und 4× schneller** als die paper-naiven $30-60 / 6-10h durch Proxy + 4-bit + LoRA-only.

---

## Trainings-Korpus: NUR fraction_ai = 1.0

User-Direktive 2026-06-15: zum Training ausschließlich Artikel verwenden, die Pangram
mit **fraction_ai = 1.0** (maximale Detection-Sicherheit) eingestuft hat.

### Verfügbarer Pool aus ki-check.db (Stand 2026-06-15)

| Outlet | n mit fraction_ai = 1.0 | n trainable (Volltext > 500 Zeichen) |
|---|---|---|
| Tagesspiegel (TSP) | 113 | 112 |
| Handelsblatt-Online (HBON) | 77 | 62 |
| Handelsblatt-Print (HB) | 74 | 59 |
| Wirtschaftswoche-Online (WWON) | 30 | 18 |
| Wirtschaftswoche-Print (WW) | 4 | 3 |
| **Total** | **325** | **254** |

### Empfohlener Split

- **200 Artikel** für Phase 1 (Proxy-Sampling) — stratifiziert nach Outlet
- **30 Artikel** für Phase 3 (GRPO-Training-Pool, kontinuierliche neue Beispiele)
- **24 Artikel** als Held-out-Eval (Phase 5, exklusiv)

Code: `src/humanizer/data/load_training_pool.py` lädt + stratifiziert.

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
