"""Phase 2 - Schritt 4: Final-Eval gegen Pangram-API + Faithfulness.

Liest Loop-Resultate (final_text), schickt Originale + Final-Texte an Pangram-API,
misst BGE-M3-Faithfulness.

Output: data/phase2/eval_results.jsonl
Kosten: 24 × 2 = 48 Pangram-Calls (~$1)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _pangram import PangramClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
LOOP = ROOT / "data" / "phase2" / "loop_results.jsonl"
OUT = ROOT / "data" / "phase2" / "eval_results.jsonl"
CACHE = ROOT / "data" / "phase2" / "pangram_cache.json"


def sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


async def main():
    if not LOOP.exists():
        sys.exit(f"LOOP fehlt: {LOOP}")
    rows = [json.loads(l) for l in LOOP.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"=== Final-Eval: {len(rows)} Loop-Resultate ===", flush=True)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # Bau Items
    items = []
    for r in rows:
        items.append((f"orig__{r['doc_id']}", r["orig_text"]))
        items.append((f"final__{r['doc_id']}", r["final_text"]))

    pangram_results: dict[str, dict] = {}
    api_items = []
    for tid, txt in items:
        h = sha(txt)
        if h in cache:
            pangram_results[tid] = cache[h]
        else:
            api_items.append({"id": tid, "text": txt, "_sha": h})

    print(f"    Cache-Hits: {len(items) - len(api_items)}/{len(items)}", flush=True)
    if api_items:
        async with PangramClient() as pc:
            res = await pc.check_bulk([{"id": it["id"], "text": it["text"]} for it in api_items])
        for it in api_items:
            r = res.get(it["id"])
            if r and not r.error and r.fraction_ai is not None:
                entry = {
                    "fraction_ai": r.fraction_ai,
                    "fraction_ai_assisted": r.fraction_ai_assisted,
                    "fraction_human": r.fraction_human,
                    "prediction": r.prediction,
                }
                cache[it["_sha"]] = entry
                pangram_results[it["id"]] = entry
            else:
                pangram_results[it["id"]] = {"error": (r.error if r else "no-result")}
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # Faithfulness
    print(f"\n--- BGE-M3 Faithfulness ---", flush=True)
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=device)

    out_rows = []
    n_bypass = 0
    n_faithful = 0
    for r in rows:
        pre = pangram_results.get(f"orig__{r['doc_id']}", {})
        post = pangram_results.get(f"final__{r['doc_id']}", {})

        # BGE-Sim
        with torch.no_grad():
            embs = enc.encode([r["orig_text"], r["final_text"]], convert_to_tensor=True,
                              normalize_embeddings=True)
            sim = (embs[0] * embs[1]).sum().item()

        pangram_pre = pre.get("fraction_ai")
        pangram_post = post.get("fraction_ai")
        bypass = (pangram_post is not None) and (pangram_post < 0.2)
        faithful = sim >= 0.85
        if bypass: n_bypass += 1
        if faithful: n_faithful += 1

        out_rows.append({
            "doc_id": r["doc_id"],
            "datum": r["datum"],
            "titel": r.get("titel"),
            "proxy_pre": r["proxy_score_pre"],
            "proxy_post": r["proxy_score_post"],
            "pangram_pre": pangram_pre,
            "pangram_post": pangram_post,
            "pangram_pred_pre": pre.get("prediction"),
            "pangram_pred_post": post.get("prediction"),
            "bge_similarity": round(sim, 3),
            "bypass_success": bypass,
            "faithful": faithful,
            "iterations": r["iterations_run"],
            "cost_usd": r.get("total_cost_usd", 0),
            "orig_chars": len(r["orig_text"]),
            "final_chars": len(r["final_text"]),
        })
        print(f"  {r['datum']} | pangram {pangram_pre} -> {pangram_post} | "
              f"sim {sim:.3f} | bypass={bypass} faithful={faithful}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_bypass_and_faithful = sum(1 for r in out_rows if r["bypass_success"] and r["faithful"])
    print(f"\n=== FINAL-EVAL ===")
    print(f"  Bypass (pangram_post < 0.2):  {n_bypass}/{len(out_rows)}")
    print(f"  Faithful (BGE-sim >= 0.85):    {n_faithful}/{len(out_rows)}")
    print(f"  BEIDE Erfolg:                  {n_bypass_and_faithful}/{len(out_rows)}")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
