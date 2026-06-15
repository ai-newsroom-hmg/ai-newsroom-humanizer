"""Phase 2 - Schritt 2: Pangram-Proxy trainieren.

BGE-M3-Embedding (1024-dim) -> 2-Layer-MLP -> fraction_ai-Score.
Trainiert auf den 200 (text, fraction_ai)-Pairs aus Schritt 1
(orig + para -> 400 Datenpunkte total).

MSE-Loss, 80/20 train/val Split. Ziel: val_MAE < 0.15.

Output: data/phase2/proxy_model.pt + proxy_config.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
PAIRS = ROOT / "data" / "phase2" / "proxy_training_pairs.jsonl"
OUT_MODEL = ROOT / "data" / "phase2" / "proxy_model.pt"
OUT_CONFIG = ROOT / "data" / "phase2" / "proxy_config.json"

EMBED_MODEL = "BAAI/bge-m3"
HIDDEN = 256
EPOCHS = 30
BATCH = 32
LR = 1e-3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class ProxyHead(nn.Module):
    def __init__(self, dim_in: int = 1024, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    if not PAIRS.exists():
        sys.exit(f"INP fehlt: {PAIRS}")

    pairs = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"=== Proxy-Training: {len(pairs)} Pairs ===", flush=True)

    # Daten: orig (label ~1.0) + para (label nach Pangram)
    samples: list[tuple[str, float]] = []
    for p in pairs:
        if p.get("orig_fraction_ai") is not None:
            samples.append((p["orig_text"], float(p["orig_fraction_ai"])))
        if p.get("para_fraction_ai") is not None:
            samples.append((p["para_text"], float(p["para_fraction_ai"])))
    print(f"    Total Samples: {len(samples)}", flush=True)

    # Label-Verteilung
    labels = [s[1] for s in samples]
    import statistics
    print(f"    Label-Mean: {statistics.mean(labels):.3f}, Stddev: {statistics.stdev(labels):.3f}",
          flush=True)
    print(f"    Histogram: 0-0.2: {sum(1 for l in labels if l < 0.2)}, "
          f"0.2-0.5: {sum(1 for l in labels if 0.2 <= l < 0.5)}, "
          f"0.5-0.8: {sum(1 for l in labels if 0.5 <= l < 0.8)}, "
          f">=0.8: {sum(1 for l in labels if l >= 0.8)}", flush=True)

    # Embeddings via BGE-M3
    print(f"\n--- BGE-M3 Embedding ({EMBED_MODEL}) ---", flush=True)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    print(f"    Device: {DEVICE}, dim: {enc.get_sentence_embedding_dimension()}", flush=True)

    texts = [s[0] for s in samples]
    embs = enc.encode(texts, batch_size=8, show_progress_bar=True, convert_to_tensor=True,
                      normalize_embeddings=True)
    embs = embs.to(DEVICE)
    y = torch.tensor(labels, dtype=torch.float32, device=DEVICE)

    # Train/Val-Split
    import random
    random.seed(42)
    idx = list(range(len(samples)))
    random.shuffle(idx)
    n_val = max(1, len(samples) // 5)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    print(f"    Split: train={len(train_idx)}, val={len(val_idx)}", flush=True)

    X_train = embs[train_idx]
    y_train = y[train_idx]
    X_val = embs[val_idx]
    y_val = y[val_idx]

    # Model
    model = ProxyHead(dim_in=embs.shape[1]).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    print(f"\n--- Training ({EPOCHS} epochs, batch {BATCH}) ---", flush=True)
    best_val_mae = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_idx))
        total_loss = 0
        n_batches = 0
        for b in range(0, len(perm), BATCH):
            batch_idx = perm[b:b+BATCH]
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1

        # Eval
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mae = (val_pred - y_val).abs().mean().item()
            val_mse = ((val_pred - y_val) ** 2).mean().item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "state_dict": model.state_dict(),
                "embed_dim": embs.shape[1],
                "hidden": HIDDEN,
            }, OUT_MODEL)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}: train_loss={total_loss/n_batches:.4f}, "
                  f"val_mae={val_mae:.4f}, val_mse={val_mse:.4f}", flush=True)

    config = {
        "embed_model": EMBED_MODEL,
        "embed_dim": embs.shape[1],
        "hidden": HIDDEN,
        "best_val_mae": best_val_mae,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "device": DEVICE,
    }
    OUT_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n=== FERTIG ===")
    print(f"  Best val_MAE: {best_val_mae:.4f}")
    print(f"  Model: {OUT_MODEL}")
    print(f"  Config: {OUT_CONFIG}")
    print(f"  Ziel <0.15 erreicht: {'JA' if best_val_mae < 0.15 else 'NEIN — Proxy zu ungenau'}")


if __name__ == "__main__":
    main()
