# ADR 004 — StealthRL-Replikation als Vergleichs-Pfad zu AuthorMist

**Status:** Geplant (Phase 2, parallel zu ADR 003)
**Datum:** 2026-06-15
**Kontext:** Recherche-Update 2026-06-15 hat StealthRL (Ranganath & Ramesh, arXiv:2602.08934, Feb 2026)
identifiziert — ein RL-basiertes Detector-Evasion-Framework mit fertigem Code und Modell-Checkpoints,
das **97,6 % Bypass auf vier Open-Source-Detectors** erreicht. **Pangram ist NICHT im Eval-Set** —
die offene Lücke ist die kommerzielle Pangram-API.

---

## Entscheidung

Statt AuthorMist von Null neu zu bauen (ADR 003), wird StealthRL als **zweite, parallel laufende
Methode** repliziert. Beide Pfade vergleichen wir gegen denselben Casdorff-Korpus + EditLens
+ Pangram-Transfer (siehe ADR 005).

## Quellen

- **Paper:** [StealthRL — arXiv:2602.08934](https://arxiv.org/abs/2602.08934) (v2 März 2026)
- **Code:** [github.com/suraj-ranganath/StealthRL](https://github.com/suraj-ranganath/StealthRL)
- **Modell-Checkpoint:** [huggingface.co/suraj-ranganath/StealthRL](https://huggingface.co/suraj-ranganath/StealthRL)

## Methode (Paper-genau)

```
Base-Model:        Qwen3-4B (HuggingFace)
LoRA-Adapter:      attached to attention + FFN projections
RL-Algorithm:      GRPO (Group Relative Policy Optimization)
Reward:            multi-detector ensemble + semantic similarity
Detectors trained against: RoBERTa, FastDetectGPT, Binoculars, MAGE
Eval-Setting:     TPR@1%FPR auf MAGE-Test-Pool (15.310 human / 14.656 AI)
```

### Reward-Komposition (StealthRL §3.2)

```
R = w_1 · (1 - mean(detector_scores))   # Evasion-Score
  + w_2 · semantic_similarity(orig, generated)  # Inhaltstreue
  + w_3 · fluency_score                  # Lesbarkeit
```

## Replikations-Setup für unseren Casdorff-Korpus

### Schritt 1 — Setup

```bash
# Repository klonen
cd ~/Projects && git clone https://github.com/suraj-ranganath/StealthRL stealthrl
cd stealthrl
pip install -e .

# Modell-Checkpoint pullen (1.5 GB)
huggingface-cli download suraj-ranganath/StealthRL --local-dir checkpoints/stealthrl-base

# Open Pangram (EditLens) als zusaetzlicher Detector
huggingface-cli download pangram/editlens_Llama-3.2-3B --local-dir checkpoints/editlens-llama-3b
huggingface-cli download pangram/editlens_roberta-large --local-dir checkpoints/editlens-roberta
```

### Schritt 2 — Inference auf vorhandenem StealthRL-Checkpoint

Erster Test: das **veröffentlichte Modell** unverändert auf unsere 24 Casdorff-Held-out-Artikel
anwenden. Wenn der Bypass schon paper-genau funktioniert, ist die Replikation einfach. Wenn nicht
(deutsch ≠ paper-Sprache, Stil-Domäne), brauchen wir Fine-Tuning auf unseren Korpus.

```python
# Quick-Smoke pseudocode
from stealthrl import StealthRLBypasser
from humanizer.data.load_training_pool import load_pool, stratified_split

bypasser = StealthRLBypasser.from_pretrained("checkpoints/stealthrl-base")
split = stratified_split(load_pool())

for art in split["eval"][:5]:
    paraphrased = bypasser.paraphrase(art["volltext"])
    print(f"{art['datum']}: {len(paraphrased)} chars")
```

### Schritt 3 — Fine-Tune auf Casdorff-Korpus (falls Schritt 2 nicht reicht)

```bash
# Setup GRPO-Training mit Casdorff-Train-Pool + multi-detector reward
python stealthrl/train.py \
    --base_model Qwen/Qwen3-4B \
    --train_data ~/Projects/ai-newsroom-humanizer/data/phase2-training-pool/train.jsonl \
    --reward_detectors editlens-llama-3b,editlens-roberta,fast-detectgpt \
    --algorithm grpo \
    --lora_rank 16 \
    --epochs 3 \
    --output_dir ~/Projects/ai-newsroom-humanizer/data/phase2/04_stealthrl_lora
```

### Schritt 4 — Eval vs. ADR-003-AuthorMist (siehe ADR 005)

## Vergleichs-Hypothesen

| Hypothese | Prediction wenn TRUE | Prediction wenn FALSE |
|---|---|---|
| StealthRL transferiert paper-genau auf Casdorff | Bypass-Rate auf EditLens > 80 % aus dem Box | < 50 %, Fine-Tune nötig |
| StealthRL-Bypass transferiert auf Pangram-API | shared architectural vulnerability bestätigt | Pangram-spezifisches Training nötig |
| AuthorMist (ADR 003, eigenes Training) > StealthRL | Detector-specific Training schlägt allgemeines | Methodisches Gleichgewicht |

## Risiken

| Risiko | Mitigation |
|---|---|
| StealthRL ist auf englische MAGE-Texte trainiert, Casdorff ist deutsch | Erst Smoke auf 5 Texten, falls Bypass <20%: Fine-Tune-Phase notwendig |
| Repository nicht aktiv gepflegt (Stand 2026-06-15) | Fork + Bug-Fixes selbst; Paper-Implementation komplett genug fuer Re-Implement |
| EditLens-Lizenz CC BY-NC-SA verbietet kommerziellen Einsatz | Forschungs-Setup OK; falls Production: Open Pangram lokal trainieren (Daten/Code released) |
| Modell-Checkpoint zu gross fuer ruediger | StealthRL ist 4B nominal, mit Qwen3-4B + LoRA-Adapter ca 8-10GB — unter 128GB Unified Memory |

## Erwartete Kosten + Zeit auf ruediger

| Phase | Kosten | Zeit |
|---|---|---|
| Schritt 1 Setup | $0 (Downloads) | 30 min |
| Schritt 2 Inference-Smoke (5 Artikel) | $0 (lokal) | 15 min |
| Schritt 3 Fine-Tune (falls noetig) | $0 (lokal) | 4-8 h MLX |
| Schritt 4 Vollerschmittels-Eval (24 Held-out + Pangram-Transfer) | $2-3 (Pangram-API) | 30 min |
| **Total** | **$2-3** | **5-9 h** |

Kein API-Pangram-Reward → keine Bottleneck. Schneller und billiger als AuthorMist-Pfad.

## Status

Skeleton-Code in `src/humanizer/training/stealthrl_replication.py` (analog grpo_proxy.py).
Setup-Script: `scripts/setup_phase2.sh`.
