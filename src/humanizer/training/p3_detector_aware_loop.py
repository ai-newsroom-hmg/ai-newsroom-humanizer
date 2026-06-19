"""Phase 2 - Schritt 3: Detector-aware Sonnet-Loop mit Proxy-Reward.

Generisches Bypass-Tool. Per Default arbeitet es auf data/phase2-training-pool/eval.jsonl,
kann aber auch fuer beliebige Input-Texte aufgerufen werden via CLI.

Pro Artikel:
  Iteration k:
    1. Sonnet generiert N=10 Varianten parallel (verschiedene temperatures)
    2. Proxy bewertet alle 10 -> beste hat niedrigsten Proxy-Score
    3. Wenn Proxy-Score < THRESHOLD: stop
    4. Sonst: beste als "current" fuer naechste Iter
  Max ITERS=5.

Stop-Kriterien:
  - proxy_score < 0.2
  - max_iter erreicht

Output: data/phase2/loop_results.jsonl
Kosten: 24 × 10 × 5 = max 1200 Sonnet-Calls (~$30), 0 Pangram-Calls (Proxy lokal)
Laufzeit: 3-6h
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanizer._openrouter import MODEL_DEFAULT, ORClient  # noqa: E402
from humanizer.core import (  # noqa: E402
    DEFAULT_MAX_ITERS,
    DEFAULT_N_VARIANTS,
    DEFAULT_THRESHOLD,
    ProxyScorer,
    humanize_text,
)

ROOT = Path(__file__).resolve().parents[3]
INP_DEFAULT = ROOT / "data" / "phase2-training-pool" / "eval.jsonl"
OUT = ROOT / "data" / "phase2" / "loop_results.jsonl"

THRESHOLD = DEFAULT_THRESHOLD
MAX_ITERS = DEFAULT_MAX_ITERS
N_VARIANTS = DEFAULT_N_VARIANTS
PARALLEL = 2
MODEL = MODEL_DEFAULT


async def loop_one(art: dict, client: ORClient, proxy: ProxyScorer, idx: int) -> dict:
    res = await humanize_text(
        art["volltext"], proxy=proxy, client=client,
        threshold=THRESHOLD, max_iters=MAX_ITERS, n_variants=N_VARIANTS,
    )
    return {
        "doc_id": art.get("doc_id"),
        "datum": art.get("datum"),
        "titel": art.get("titel"),
        "quelle": art.get("quelle"),
        "orig_text": res.orig_text,
        "final_text": res.final_text,
        "proxy_score_pre": res.proxy_score_pre,
        "proxy_score_post": res.proxy_score_post,
        "iterations_run": res.iterations_run,
        "history": res.history,
        "total_cost_usd": res.total_cost_usd,
        "duration_s": res.duration_s,
    }


async def main():
    inp = Path(os.environ.get("HUMANIZER_INPUT", str(INP_DEFAULT)))
    if not inp.exists():
        sys.exit(f"INP fehlt: {inp}")
    arts = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Erwarte 'volltext' Feld (DB-Schema) — falls 'text': renaming
    for a in arts:
        if "volltext" not in a:
            a["volltext"] = a.get("text") or a.get("body") or a.get("content") or ""

    print(f"=== Detector-aware Loop: {len(arts)} Artikel ===", flush=True)
    print(f"    THRESHOLD: {THRESHOLD}, MAX_ITERS: {MAX_ITERS}, N_VARIANTS: {N_VARIANTS}", flush=True)

    print("\n--- Proxy laden ---", flush=True)
    proxy = ProxyScorer()
    print(f"    Best val_MAE: {proxy.config.get('best_val_mae', '?'):.4f}", flush=True)

    client = ORClient(model=MODEL)
    sem = asyncio.Semaphore(PARALLEL)
    results = [None] * len(arts)
    done = 0
    total_cost = 0.0
    t_start = time.time()

    async def worker(i, art):
        nonlocal done, total_cost
        async with sem:
            r = await loop_one(art, client, proxy, i)
            results[i] = r
            done += 1
            total_cost += r.get("total_cost_usd", 0)
            elapsed = time.time() - t_start
            pre = r["proxy_score_pre"]
            post = r["proxy_score_post"]
            tag = "OK" if post < THRESHOLD else "FAIL"
            print(f"  [{done}/{len(arts)}] {tag:>4} {art.get('datum','?')} "
                  f"proxy {pre:.3f} -> {post:.3f} "
                  f"({r['iterations_run']} iter, {r['duration_s']:.0f}s, "
                  f"${r['total_cost_usd']:.3f}) "
                  f"[total ${total_cost:.2f}, elapsed {elapsed:.0f}s]", flush=True)

    await asyncio.gather(*(worker(i, a) for i, a in enumerate(arts)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_proxy_success = sum(1 for r in results if r["proxy_score_post"] < THRESHOLD)
    print("\n=== FERTIG ===")
    print(f"  Proxy-Bypass (proxy < {THRESHOLD}): {n_proxy_success}/{len(results)}")
    print(f"  Total Sonnet-Kosten: ${total_cost:.4f}")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
