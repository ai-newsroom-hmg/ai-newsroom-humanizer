"""Phase 3b Nightly-Run (2026-06-18 → 2026-06-19).

Über-Nacht-Run für morgen-früh-Demo:
- 12 Casdorff-Phase3a-Docs × Best-of-50 (Mistral-3.2-instruct) — Härtefall (alle pre=1.0)
- 6 OOD-Docs × Best-of-30 — Generalisierungs-Probe
- 1 Demo-Target (8675 chars) × Best-of-24 chunked — saubere Demo-Variante

BGE-Filter ≥ 0.85 VOR Pangram (Cost-Save vs Phase 3a).
Pangram-Cache (data/phase2/pangram_cache.json, 776 entries) wiederverwenden.

Output:
  data/phase3b/nightly_results.jsonl       — Variant-Level (alle Mistral-Calls)
  data/phase3b/nightly_summary.json        — Per-Doc-Summary
  data/phase3b/nightly_target_final.txt    — Best-Variant des Demo-Texts
  data/phase3b/nightly.log                 — Live-Log

Cost-Estimate: $30-40 (Pangram dominiert).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from humanizer._openrouter import ORClient  # noqa: E402
from humanizer._pangram import PangramClient  # noqa: E402
from humanizer.core_bestofn import (  # noqa: E402
    MODEL_MISTRAL_3_2,
    SYSTEM_PROMPT_BESTOFN,
    USER_PROMPT,
    bge_similarity_batch,
    humanize_chunked_bestofn,
)

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "data" / "phase2-training-pool" / "eval.jsonl"
SAMPLE_PHASE3A = ROOT / "data" / "phase3a" / "sample_n12.jsonl"
TARGET_TXT = ROOT / "data" / "test-corpora" / "target-text-2026-06-18-journalismus-tot.txt"
CACHE = ROOT / "data" / "phase2" / "pangram_cache.json"

OUT_DIR = ROOT / "data" / "phase3b"
OUT_VARIANTS = OUT_DIR / "nightly_results.jsonl"
OUT_SUMMARY = OUT_DIR / "nightly_summary.json"
OUT_TARGET = OUT_DIR / "nightly_target_final.txt"
OUT_TARGET_TRACE = OUT_DIR / "nightly_target_trace.json"

# Best-of-50 für Casdorff: 10 Temps × 5 idx
TEMPS_50 = (0.85, 0.88, 0.92, 0.95, 0.98, 1.00, 1.03, 1.07, 1.10, 1.15)
N_IDX_50 = 5
# Best-of-30 für OOD: 6 Temps × 5 idx
TEMPS_30 = (0.88, 0.95, 1.00, 1.05, 1.10, 1.15)
N_IDX_30 = 5

BGE_THRESHOLD = 0.85


def sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def pick_ood_sample(docs_all: list[dict], exclude_ids: set[str], n_per_bucket=(2, 2, 2)) -> list[dict]:
    """6 OOD-Docs: 2 short / 2 mid / 2 long, fraction_ai=1.0, NICHT in Casdorff-Sample."""
    elig = [d for d in docs_all
            if d.get("fraction_ai") == 1.0
            and len(d.get("volltext", "")) >= 500
            and d["doc_id"] not in exclude_ids
            and not d["doc_id"].startswith(("TSP__074e1b8d", "WWON__45d50bf9", "HB_100214957"))]
    short = sorted([d for d in elig if len(d["volltext"]) <= 1400], key=lambda x: len(x["volltext"]))
    mid = sorted([d for d in elig if 1400 < len(d["volltext"]) <= 2800],
                 key=lambda x: len(x["volltext"]))
    lng = sorted([d for d in elig if len(d["volltext"]) > 2800], key=lambda x: len(x["volltext"]))
    log(f"[ood-pool] short={len(short)} mid={len(mid)} long={len(lng)}")
    return short[:n_per_bucket[0]] + mid[:n_per_bucket[1]] + lng[:n_per_bucket[2]]


async def gen_one(client: ORClient, text: str, temp: float, idx: int, max_tokens: int) -> dict:
    try:
        out = await client.complete(SYSTEM_PROMPT_BESTOFN, USER_PROMPT.format(text=text),
                                    temperature=temp, max_tokens=max_tokens)
        return {"text": out["text"].strip(), "cost": out["cost_usd"],
                "temp": temp, "idx": idx, "error": None}
    except Exception as e:
        return {"text": "", "cost": 0.0, "temp": temp, "idx": idx, "error": str(e)[:200]}


async def gen_variants_for_doc(client: ORClient, orig: str, temps: tuple, n_idx: int) -> list[dict]:
    max_tokens = max(3000, int(len(orig) * 0.6))
    tasks = [gen_one(client, orig, t, k, max_tokens) for t in temps for k in range(n_idx)]
    return await asyncio.gather(*tasks)


def bge_filter(orig: str, variants: list[dict], threshold: float) -> None:
    """In-place: schreibt bge_sim + faithful Flag."""
    texts = [v["text"] for v in variants if not v["error"] and v["text"]]
    if not texts:
        return
    sims = bge_similarity_batch(orig, texts)
    j = 0
    for v in variants:
        if not v["error"] and v["text"]:
            v["bge_sim"] = sims[j]
            v["faithful"] = sims[j] >= threshold
            j += 1
        else:
            v["bge_sim"] = 0.0
            v["faithful"] = False


async def pangram_eval(items_to_check: list[dict], cache: dict) -> dict:
    """items_to_check = [{"id": ..., "text": ..., "sha": ...}]
    Cache-Hit-first, dann API. Returns: {id: {fraction_ai, prediction}}.
    """
    out = {}
    api_items = []
    for it in items_to_check:
        if it["sha"] in cache:
            out[it["id"]] = cache[it["sha"]]
        else:
            api_items.append(it)
    if not api_items:
        return out
    log(f"  pangram: {len(items_to_check)} total, {len(out)} cached, {len(api_items)} API "
        f"-> ${len(api_items) * 0.05:.2f}")
    async with PangramClient() as pc:
        res = await pc.check_bulk([{"id": it["id"], "text": it["text"]} for it in api_items])
    for it in api_items:
        r = res.get(it["id"])
        if r and not r.error and r.fraction_ai is not None:
            entry = {"fraction_ai": r.fraction_ai,
                     "fraction_human": r.fraction_human,
                     "prediction": r.prediction}
            cache[it["sha"]] = entry
            out[it["id"]] = entry
        else:
            out[it["id"]] = {"error": (r.error if r else "no-result")}
    return out


async def run_doc(client: ORClient, doc: dict, kind: str, temps: tuple, n_idx: int,
                  cache: dict, out_f, summary: dict) -> None:
    did = doc["doc_id"]
    orig = doc["volltext"]
    n_total = len(temps) * n_idx
    log(f"\n=== {kind} | {did[:30]:<30} ({len(orig)} chars) - Best-of-{n_total} ===")
    t0 = time.time()
    variants = await gen_variants_for_doc(client, orig, temps, n_idx)
    n_ok = sum(1 for v in variants if not v["error"])
    gen_cost = sum(v["cost"] for v in variants)
    log(f"  gen ok: {n_ok}/{n_total} in {time.time() - t0:.0f}s, ${gen_cost:.4f}")

    bge_filter(orig, variants, BGE_THRESHOLD)
    n_faithful = sum(1 for v in variants if v.get("faithful"))
    log(f"  BGE-faithful (>={BGE_THRESHOLD}): {n_faithful}/{n_ok}")

    pangram_items = [{"id": f"{did}__t{v['temp']}__i{v['idx']}",
                      "text": v["text"], "sha": sha(v["text"])}
                     for v in variants if v.get("faithful")]
    if pangram_items:
        results = await pangram_eval(pangram_items, cache)
        idx_map = {f"{did}__t{v['temp']}__i{v['idx']}": v for v in variants}
        for k, r in results.items():
            v = idx_map.get(k)
            if v and "fraction_ai" in r:
                v["pangram_fraction_ai"] = r["fraction_ai"]
                v["pangram_prediction"] = r["prediction"]

    bypass_vs = [v for v in variants
                 if v.get("faithful") and v.get("pangram_fraction_ai", 1.0) < 0.2]
    n_bypass = len(bypass_vs)

    if bypass_vs:
        best = min(bypass_vs, key=lambda v: v["pangram_fraction_ai"])
        stop = "ranked_by_pangram_bypass"
    else:
        faithful_only = [v for v in variants if v.get("faithful")
                         and v.get("pangram_fraction_ai") is not None]
        if faithful_only:
            best = min(faithful_only, key=lambda v: v["pangram_fraction_ai"])
            stop = "no_bypass_best_faithful"
        else:
            ok = [v for v in variants if not v["error"]]
            best = max(ok, key=lambda v: v.get("bge_sim", 0.0)) if ok else None
            stop = "no_faithful_fallback_best_bge"

    log(f"  faithful={n_faithful} bypass={n_bypass} "
        f"best_pangram={best.get('pangram_fraction_ai') if best else 'n/a'} "
        f"best_bge={best.get('bge_sim') if best else 'n/a'} stop={stop}")

    for v in variants:
        v["doc_id"] = did
        v["kind"] = kind
        v["orig_chars"] = len(orig)
        out_f.write(json.dumps(v, ensure_ascii=False) + "\n")
        out_f.flush()

    summary[did] = {
        "kind": kind, "chars": len(orig), "autor": doc.get("autor"),
        "n_total": n_total, "n_ok": n_ok, "n_faithful": n_faithful, "n_bypass": n_bypass,
        "best_pangram_fraction_ai": best.get("pangram_fraction_ai") if best else None,
        "best_bge_sim": best.get("bge_sim") if best else None,
        "stopped_reason": stop,
        "gen_cost_usd": round(gen_cost, 4),
        "duration_s": round(time.time() - t0, 1),
    }
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_target_chunked(client: ORClient, cache: dict, summary: dict) -> None:
    """Demo-Target: Best-of-24 chunked, finale Variante schreiben."""
    log("\n=== TARGET | chunked Best-of-24 ===")
    orig = TARGET_TXT.read_text()
    t0 = time.time()

    res = await humanize_chunked_bestofn(
        orig, client=client,
        n_variants_per_chunk=24,
        bge_threshold=0.80,
        rank_by_pangram=True,
        on_progress=lambda m: log(f"  {m}"),
    )
    OUT_TARGET.write_text(res.final_text, encoding="utf-8")
    OUT_TARGET_TRACE.write_text(json.dumps({
        "orig_chars": len(orig),
        "final_chars": len(res.final_text),
        "bge_sim_global": res.bge_sim,
        "n_generated": res.n_generated,
        "n_faithful": res.n_faithful,
        "duration_s": res.duration_s,
        "cost_usd": res.total_cost_usd,
        "chunk_results": res.chunk_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"  TARGET done: bge={res.bge_sim:.3f} n_faithful={res.n_faithful} "
        f"chars={len(res.final_text)} ({time.time() - t0:.0f}s, ${res.total_cost_usd:.3f})")
    log(f"  -> final: {OUT_TARGET}")

    summary["__TARGET__"] = {
        "kind": "demo_target_chunked",
        "orig_chars": len(orig),
        "final_chars": len(res.final_text),
        "bge_sim_global": res.bge_sim,
        "n_generated": res.n_generated,
        "n_faithful": res.n_faithful,
        "duration_s": res.duration_s,
        "cost_usd": res.total_cost_usd,
        "chunk_results": res.chunk_results,
    }


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== Nightly Phase-3b Run startet ===")

    # Cache laden
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    log(f"Pangram-Cache: {len(cache)} entries")

    # Casdorff sample
    casdorff_ids = [json.loads(l)["doc_id"]
                    for l in SAMPLE_PHASE3A.read_text().splitlines() if l.strip()]
    log(f"Casdorff sample: {len(casdorff_ids)} doc_ids")
    docs_all = [json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()]
    docs_map = {d["doc_id"]: d for d in docs_all}
    casdorff_docs = [docs_map[i] for i in casdorff_ids if i in docs_map]
    log(f"Casdorff docs hydriert: {len(casdorff_docs)}")

    # OOD pick
    ood_docs = pick_ood_sample(docs_all, exclude_ids=set(casdorff_ids))
    log(f"OOD sample: {len(ood_docs)} docs")
    for d in ood_docs:
        log(f"  OOD: {d['doc_id'][:30]:<30} {len(d['volltext']):>5} chars  {(d.get('autor') or '?')[:30]}")

    client = ORClient(model=MODEL_MISTRAL_3_2)
    summary: dict = {}
    t_start = time.time()

    with OUT_VARIANTS.open("w", encoding="utf-8") as out_f:
        log("\n>>> PHASE 1: Casdorff Best-of-50")
        for d in casdorff_docs:
            await run_doc(client, d, "casdorff", TEMPS_50, N_IDX_50, cache, out_f, summary)

        log("\n>>> PHASE 2: OOD Best-of-30")
        for d in ood_docs:
            await run_doc(client, d, "ood", TEMPS_30, N_IDX_30, cache, out_f, summary)

        log("\n>>> PHASE 3: Demo-Target chunked Best-of-24")
        await run_target_chunked(client, cache, summary)

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    log(f"\n=== Nightly DONE in {elapsed / 60:.1f} min ===")
    log("Output:")
    log(f"  {OUT_VARIANTS}")
    log(f"  {OUT_SUMMARY}")
    log(f"  {OUT_TARGET}")

    # Print results table
    log("\n=== RESULTATE ===")
    log(f"{'doc_id':<32} {'kind':<10} {'chars':>5} {'n_faith':>7} {'n_byp':>6} {'best_p':>7} {'best_bge':>9}")
    bypass_docs = {"casdorff": 0, "ood": 0}
    total_docs = {"casdorff": 0, "ood": 0}
    for did, s in summary.items():
        if did == "__TARGET__":
            continue
        bp = s["best_pangram_fraction_ai"]
        bp_s = f"{bp:.3f}" if isinstance(bp, (int, float)) else "n/a"
        bg = s.get("best_bge_sim", 0.0)
        bg_s = f"{bg:.3f}" if isinstance(bg, (int, float)) else "n/a"
        log(f"{did[:32]:<32} {s['kind']:<10} {s['chars']:>5} {s['n_faithful']:>7} "
            f"{s['n_bypass']:>6} {bp_s:>7} {bg_s:>9}")
        total_docs[s["kind"]] += 1
        if s["n_bypass"] >= 1:
            bypass_docs[s["kind"]] += 1
    for k in ("casdorff", "ood"):
        if total_docs[k]:
            log(f"\n>>> {k}: Doc-Bypass {bypass_docs[k]}/{total_docs[k]} "
                f"= {bypass_docs[k] / total_docs[k] * 100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
