"""Phase 2 Master-Orchestrator — AuthorMist-RL mit Proxy-Reward auf ruediger.

5 Phasen, alle einzeln aufrufbar via --phase {1..5}:

  1  initial_sampling     180 Artikel × Sonnet-Paraphrase → 180 (text, pangram_score)
                          Output: data/phase2/01_proxy_training_data.jsonl
                          Kosten: ~$4 (180 Pangram-Calls)
                          Lauf:    30 min

  2  train_proxy          BGE-M3 Embedding + 2-Layer-MLP → Pangram-Score-Vorhersage
                          Output: data/phase2/02_proxy_model.safetensors
                          Kosten: $0 (lokal MLX)
                          Lauf:    10 min auf ruediger

  3  grpo_train           Qwen2.5-3B-4bit + LoRA + GRPO, Proxy als Reward
                          Output: data/phase2/03_humanizer_lora.safetensors
                          Kosten: $0 (Proxy)
                          Lauf:    1-2 h auf ruediger
                          Drift-Check: alle 100 steps 1 echter Pangram-Call (~$0.40)

  4  eval_holdout         24 Artikel × Humanizer-Forward + Pangram pre/post
                          Output: data/phase2/04_eval_results.jsonl
                          Kosten: ~$2 (100 Pangram-Calls)
                          Lauf:    20 min

  5  report               Excel + Word + Phase-2-Befund-ADR (Update fuer 003)
                          Output: ~/Downloads/ai-newsroom-humanizer/phase2-<date>/
                          Kosten: $0
                          Lauf:    5 min

GESAMT: ~$6-8 Pangram + 2-3 h Compute auf ruediger (M5 Max 128 GB).

WICHTIG (Pflichtdoku siehe ADR 003):
- Training-Pool: NUR Artikel mit fraction_ai = 1.0 (User-Direktive 2026-06-15)
- ruediger = default fuer alle KI-Compute (Memory: ruediger-default-for-ki-compute)
- Pangram-Reward-Caching: text_sha256 -> score (data/phase2/pangram_cache.json)
- Budget-Guard: Pangram-Calls insgesamt < 500 (Hard-Stop via env PANGRAM_HARD_BUDGET)

Status: SKELETON. Echte Implementierung wird gestartet wenn Phase-2-Setup auf
ruediger steht (MLX-GRPO installiert, Qwen2.5-3B-4bit gecached).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
PHASE2_DIR = PROJ / "data" / "phase2"
PHASE2_DIR.mkdir(parents=True, exist_ok=True)


# ─── Phase 1: Initial Sampling ──────────────────────────────────────────────
def phase_01_initial_sampling():
    """180 Artikel laden -> Sonnet paraphrasiert sie -> Pangram bewertet beide.

    Ergebnis: gepaarte Trainings-Daten (text, pangram_score) fuer Proxy-Modell.
    Wir nehmen das Original (fraction_ai=1.0) UND die paraphrasierte Version
    (vermutlich 0.5-1.0) als breitestes Pangram-Score-Spektrum.

    TODO (echte Implementation):
      from humanizer.data.load_training_pool import load_pool, stratified_split
      from humanizer._openrouter import ORClient
      from ki_check.pangram import PangramClient

      pool = load_pool()
      split = stratified_split(pool)
      proxy_items = split["proxy"]   # 180

      # Sonnet paraphrasiert jeden Artikel (mit cost-cap)
      # Pangram bewertet Original + Paraphrase
      # Speichern als JSONL: {doc_id, text_variant, fraction_ai, source}
    """
    print("[Phase 1] STUB — wird implementiert wenn ruediger-Setup steht")
    print("          Erwartet 360 Pangram-Calls (180 Orig + 180 Sonnet-Paraphrasen) = ~$7")
    print(f"          Output: {PHASE2_DIR / '01_proxy_training_data.jsonl'}")


# ─── Phase 2: Proxy-Modell ──────────────────────────────────────────────────
def phase_02_train_proxy():
    """BGE-M3 Embedding (XLM-RoBERTa) + 2-Layer-MLP -> Score 0..1.

    Loss: MSE gegen echte Pangram-fraction_ai.
    Lokal auf ruediger MPS — sentence-transformers ist MPS-ready.

    TODO:
      from sentence_transformers import SentenceTransformer
      import torch, torch.nn as nn

      enc = SentenceTransformer('BAAI/bge-m3', device='mps')
      X = enc.encode(texts, convert_to_tensor=True)
      class Proxy(nn.Module): ...   # 1024 -> 256 -> 1 (sigmoid)
      train_loop(Proxy, X, y_pangram, epochs=20)
      save(model, PHASE2_DIR / '02_proxy_model.safetensors')

    Eval: 80/20 train/val split innerhalb der 360 Texte.
    Target: val MAE < 0.10 (gut genug als Reward-Proxy).
    """
    print("[Phase 2] STUB — sentence-transformers BGE-M3 + MLP")
    print("          Erwartet 0 Pangram-Calls (lokal)")
    print(f"          Output: {PHASE2_DIR / '02_proxy_model.safetensors'}")


# ─── Phase 3: GRPO-Training ─────────────────────────────────────────────────
def phase_03_grpo_train():
    """Qwen2.5-3B-Instruct-4bit + LoRA + GRPO mit Proxy-Reward.

    Stack:
      mlx-community/Qwen2.5-3B-Instruct-4bit  (Apple-Silicon-native)
      mlx-lm-lora                             (LoRA-only-Training)
      mlx-tune oder MLX-GRPO                  (GRPO-Algorithmus)

    Pro Training-Step:
      1. Sample 8-16 train-pool Artikel
      2. Modell generiert N=4 Varianten je Artikel
      3. Proxy-Modell bewertet jede Variante -> reward
      4. Optional: BGE-M3-Similarity(orig, generated) > 0.94 als secondary reward
      5. GRPO Policy-Update

    Drift-Check alle 100 Steps:
      1 Variant -> echte Pangram-API -> Validierung dass Proxy nicht drifted

    Stop:
      - max_steps = 300, oder
      - moving-avg reward > 0.85, oder
      - manual stop

    Output: LoRA-Adapter unter data/phase2/03_humanizer_lora/.
    """
    print("[Phase 3] STUB — MLX-GRPO + Qwen2.5-3B-4bit + LoRA")
    print("          Erwartet ~20 Pangram-Drift-Calls = ~$0.40")
    print("          Compute: 1-2 h auf ruediger M5 Max 128 GB")
    print(f"          Output: {PHASE2_DIR / '03_humanizer_lora'}")


# ─── Phase 4: Held-out Evaluation ──────────────────────────────────────────
def phase_04_eval_holdout():
    """24 Held-out-Artikel durch den Humanizer + Pangram pre/post.

    Pro Artikel:
      pre  = fraction_ai (1.0, weil train-pool-Filter)
      post = Pangram(Humanizer(text))

    Plus Faithfulness-Check (BGE-M3-Similarity > 0.94 erforderlich).

    Metriken:
      - Bypass-Rate     = % mit fraction_ai_post < 0.2
      - Mean drop       = mean(pre - post)
      - Faithfulness    = % mit sim > 0.94
      - vs Phase-1-Baseline (Strategie 02: 4%, 03: 0%, A: 0%, 05: 40% mit Inhalt-Breach)
    """
    print("[Phase 4] STUB — Held-out Eval (24 Artikel exklusiv)")
    print("          Erwartet 100 Pangram-Calls = ~$2")
    print(f"          Output: {PHASE2_DIR / '04_eval_results.jsonl'}")


# ─── Phase 5: Report ──────────────────────────────────────────────────────
def phase_05_report():
    """Excel + Word-Files + ADR-Update analog Phase 1.

    Wiederverwendet humanizer.export.
    """
    print("[Phase 5] STUB — Excel + Word + ADR-Update")
    print("          Output: ~/Downloads/ai-newsroom-humanizer/phase2-<date>/")


# ─── CLI ───────────────────────────────────────────────────────────────────
PHASES = {
    1: ("initial_sampling", phase_01_initial_sampling),
    2: ("train_proxy",      phase_02_train_proxy),
    3: ("grpo_train",       phase_03_grpo_train),
    4: ("eval_holdout",     phase_04_eval_holdout),
    5: ("report",           phase_05_report),
}


def main():
    ap = argparse.ArgumentParser(description="Phase 2 GRPO-Proxy Orchestrator")
    ap.add_argument("--phase", type=int, choices=PHASES.keys(),
                    help="Einzel-Phase laufen (sonst: alle 1-5 sequenziell)")
    ap.add_argument("--pangram-budget", type=float, default=15.0,
                    help="Maximale Pangram-Kosten in USD (default 15)")
    args = ap.parse_args()

    os.environ["PANGRAM_HARD_BUDGET"] = str(args.pangram_budget)
    print(f"=== Phase 2 Orchestrator — Pangram-Budget ${args.pangram_budget} ===\n")

    if args.phase:
        name, fn = PHASES[args.phase]
        print(f"Lauf Phase {args.phase}: {name}\n")
        fn()
    else:
        for num, (name, fn) in sorted(PHASES.items()):
            print(f"\n=== Phase {num}: {name} ===")
            fn()


if __name__ == "__main__":
    main()
