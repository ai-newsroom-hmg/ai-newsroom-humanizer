#!/usr/bin/env bash
# Phase-2-Setup fuer ruediger (M5 Max 128 GB MLX-Stack)
#
# Bereitet vor:
#   - StealthRL Repo (github.com/suraj-ranganath/StealthRL) cloned + installiert
#   - StealthRL Modell-Checkpoint von HuggingFace gepullt (~1.5 GB)
#   - Open Pangram (EditLens) cloned + Checkpoints gepullt
#       - editlens_Llama-3.2-3B (~6 GB)
#       - editlens_roberta-large (~1.4 GB)
#   - mlx-lm-lora installiert (12 RL-Algorithms inkl. GRPO/GSPO/Dr.GRPO/DAPO)
#   - Qwen3-4B Base-Model gecached (~8 GB) — fuer eigenes AuthorMist-Training
#
# Vorbedingung:
#   - ruediger erreichbar via SSH (Tailscale)
#   - Python 3.11+, pip mit --user/--break-system-packages
#   - HF-Token in ~/.cache/huggingface/token (falls gated repos)
#
# Zeit: ~30 min Setup, ~20 GB Disk

set -euo pipefail

PROJ_HOME="$HOME/Projects"
EXTERNAL_DIR="$PROJ_HOME/ai-newsroom-humanizer/data/external"

echo "=== Phase-2-Setup: StealthRL + Open Pangram + AuthorMist-Stack ==="
echo "Target Host: $(hostname)"
echo "Working Dir: $EXTERNAL_DIR"
echo

mkdir -p "$EXTERNAL_DIR"

# ─── 1. StealthRL ───────────────────────────────────────────────────────────
echo "--- 1. StealthRL Repo + Checkpoint ---"
if [ ! -d "$PROJ_HOME/stealthrl" ]; then
    git clone https://github.com/suraj-ranganath/StealthRL "$PROJ_HOME/stealthrl"
else
    echo "  Repo schon da, pull..."
    git -C "$PROJ_HOME/stealthrl" pull --rebase || true
fi

cd "$PROJ_HOME/stealthrl"
pip install --user --break-system-packages -e . 2>&1 | tail -5 || \
    echo "  WARN: pip install -e failed, lese requirements manuell"

# Checkpoint pullen via HF CLI
if ! command -v huggingface-cli >/dev/null; then
    pip install --user --break-system-packages -U huggingface_hub
fi

CKPT_DIR="$EXTERNAL_DIR/stealthrl-base"
if [ ! -d "$CKPT_DIR" ]; then
    huggingface-cli download suraj-ranganath/StealthRL --local-dir "$CKPT_DIR"
else
    echo "  StealthRL-Checkpoint vorhanden: $CKPT_DIR"
fi

# ─── 2. Open Pangram (EditLens) ────────────────────────────────────────────
echo
echo "--- 2. Open Pangram (EditLens) Models ---"

if [ ! -d "$PROJ_HOME/EditLens" ]; then
    git clone https://github.com/pangramlabs/EditLens "$PROJ_HOME/EditLens"
else
    git -C "$PROJ_HOME/EditLens" pull --rebase || true
fi

for model in "editlens_Llama-3.2-3B" "editlens_roberta-large"; do
    local_dir="$EXTERNAL_DIR/$model"
    if [ ! -d "$local_dir" ]; then
        echo "  Pull $model -> $local_dir"
        huggingface-cli download "pangram/$model" --local-dir "$local_dir" || \
            echo "  WARN: Pull failed for $model — check HF access"
    else
        echo "  $model schon da"
    fi
done

# EditLens training dataset (optional, ~500 MB)
DATA_DIR="$EXTERNAL_DIR/editlens_iclr_dataset"
if [ ! -d "$DATA_DIR" ]; then
    huggingface-cli download pangram/editlens_iclr --repo-type dataset --local-dir "$DATA_DIR" || \
        echo "  Skip dataset (optional)"
fi

# ─── 3. mlx-lm-lora (GRPO + Varianten) ─────────────────────────────────────
echo
echo "--- 3. mlx-lm-lora installieren ---"
pip install --user --break-system-packages -U mlx-lm-lora 2>&1 | tail -3 || true

python3 -c "import mlx_lm_lora; print(f'mlx-lm-lora OK: {mlx_lm_lora.__version__}')" || \
    echo "  WARN: mlx-lm-lora import failed"

# ─── 4. Qwen3-4B Base-Model (fuer eigenes Training, ADR 003) ───────────────
echo
echo "--- 4. Qwen3-4B-Instruct-4bit cachen ---"
QWEN_DIR="$EXTERNAL_DIR/Qwen3-4B-Instruct-4bit"
if [ ! -d "$QWEN_DIR" ]; then
    huggingface-cli download mlx-community/Qwen3-4B-Instruct-4bit --local-dir "$QWEN_DIR" || \
        huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir "$EXTERNAL_DIR/Qwen2.5-3B-Instruct"
else
    echo "  Qwen3 schon da"
fi

# ─── 5. BGE-M3 fuer Faithfulness-Embedding ──────────────────────────────────
echo
echo "--- 5. BGE-M3 Embedding-Modell ---"
pip install --user --break-system-packages -U sentence-transformers 2>&1 | tail -2 || true

python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-m3')
print('BGE-M3 cached:', m.get_sentence_embedding_dimension(), 'dim')
" || echo "  WARN: BGE-M3 setup failed"

# ─── Health-Check ──────────────────────────────────────────────────────────
echo
echo "=== Health-Check ==="
echo "Disk usage external/:"
du -sh "$EXTERNAL_DIR"/* 2>/dev/null | sort -h
echo
echo "Verfuegbare Stacks:"
python3 -c "
import importlib
for mod in ('mlx', 'mlx_lm', 'mlx_lm_lora', 'sentence_transformers', 'transformers'):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, '__version__', '?')
        print(f'  OK  {mod} = {v}')
    except ImportError as e:
        print(f'  --  {mod} fehlt ({e})')
"

echo
echo "=== Setup fertig. Naechster Schritt: ==="
echo "  cd ~/Projects/ai-newsroom-humanizer"
echo "  python src/humanizer/training/grpo_proxy.py --phase 1      # AuthorMist-Pfad (ADR 003)"
echo "  python src/humanizer/training/stealthrl_smoke.py           # StealthRL-Pfad (ADR 004)"
