"""Phase 2 - Schritt 1: Proxy-Trainings-Daten sammeln.

Pro Train-Artikel:
  1. Originaltext (fraction_ai = 1.0 in DB)
  2. Sonnet-Paraphrase via OpenRouter
  3. Beide an Pangram-API -> echte fraction_ai-Scores
  4. Speichere als (text, fraction_ai) Pair

Mit Pangram-SHA256-Cache: keine Doppel-Calls.

Output: data/phase2/proxy_training_pairs.jsonl
Kosten: 200 Sonnet (~$3) + 400 Pangram (~$8) = ~$11
Laufzeit: 2-3 h
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _openrouter import ORClient, MODEL_DEFAULT  # noqa: E402

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
from humanizer._pangram import PangramClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TRAIN_POOL = ROOT / "data" / "phase2-training-pool" / "proxy.jsonl"
OUT = ROOT / "data" / "phase2" / "proxy_training_pairs.jsonl"
CACHE = ROOT / "data" / "phase2" / "pangram_cache.json"

# Limits
N_ARTICLES = int(os.environ.get("N_ARTICLES", "200"))
PARALLEL_SONNET = 4
PARALLEL_PANGRAM = 8

PARAPHRASE_SYSTEM = """Du bist ein erfahrener deutscher Journalist. Paraphrasiere
den folgenden politischen Kommentar oder Artikel auf Deutsch — Inhalt bleibt
exakt gleich, Stil variiert (andere Saetze, andere Konnektoren, andere
Reihenfolge). Antworte NUR mit dem paraphrasierten Text. Keine Vorrede."""

PARAPHRASE_USER = "Originaltext:\n\n{text}\n\nParaphrasierte Version:"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


async def paraphrase_one(client: ORClient, text: str) -> tuple[str, float]:
    out = await client.complete(
        PARAPHRASE_SYSTEM,
        PARAPHRASE_USER.format(text=text),
        temperature=0.7, max_tokens=3000,
    )
    return out["text"].strip(), out["cost_usd"]


async def pangram_batch(pc: PangramClient, texts: list[tuple[str, str]], cache: dict) -> dict[str, dict]:
    """texts = [(id, text), ...] -> {id: {fraction_ai, fraction_human, prediction}}.
    Cache-aware: gleicher Text-SHA -> kein API-Call."""
    new_items = []
    out: dict[str, dict] = {}
    for tid, txt in texts:
        h = sha(txt)
        if h in cache:
            out[tid] = cache[h]
        else:
            new_items.append({"id": tid, "text": txt, "_sha": h})

    if new_items:
        res = await pc.check_bulk([{"id": it["id"], "text": it["text"]} for it in new_items])
        for it in new_items:
            r = res.get(it["id"])
            if r and not r.error and r.fraction_ai is not None:
                entry = {
                    "fraction_ai": r.fraction_ai,
                    "fraction_ai_assisted": r.fraction_ai_assisted,
                    "fraction_human": r.fraction_human,
                    "prediction": r.prediction,
                }
                cache[it["_sha"]] = entry
                out[it["id"]] = entry
            else:
                out[it["id"]] = {"error": (r.error if r else "no-result")}
    return out


async def main():
    arts = [json.loads(l) for l in TRAIN_POOL.read_text(encoding="utf-8").splitlines() if l.strip()]
    arts = arts[:N_ARTICLES]
    print(f"=== Proxy-Daten-Sammlung: {len(arts)} Artikel ===", flush=True)

    cache = load_cache()
    print(f"    Cache: {len(cache)} Eintraege", flush=True)

    or_client = ORClient(model=MODEL_DEFAULT)

    t_start = time.time()
    done = 0
    total_sonnet_cost = 0.0
    pairs: list[dict] = []

    # Sonnet-Paraphrasen parallel
    print(f"--- Phase 1.1: Sonnet-Paraphrasen ({PARALLEL_SONNET}-way) ---", flush=True)
    sem = asyncio.Semaphore(PARALLEL_SONNET)

    async def gen_para(i: int, art: dict):
        nonlocal done, total_sonnet_cost
        async with sem:
            try:
                para, cost = await paraphrase_one(or_client, art["volltext"])
                total_sonnet_cost += cost
                done += 1
                if done % 10 == 0 or done == len(arts):
                    print(f"  paraphrased {done}/{len(arts)} (cost ${total_sonnet_cost:.2f}, "
                          f"elapsed {time.time()-t_start:.0f}s)", flush=True)
                return {"doc_id": art["doc_id"], "orig_text": art["volltext"], "para_text": para,
                        "datum": art["datum"], "quelle": art["quelle"], "sonnet_cost": cost}
            except Exception as e:
                print(f"  FAIL {art['doc_id'][:16]}: {e!r}", flush=True)
                return None

    para_results = await asyncio.gather(*(gen_para(i, a) for i, a in enumerate(arts)))
    para_results = [r for r in para_results if r]
    print(f"  -> {len(para_results)} Paraphrasen, total ${total_sonnet_cost:.2f}", flush=True)

    # Pangram-Scores: Originale + Paraphrasen
    print(f"\n--- Phase 1.2: Pangram-Scores (Cache-aware) ---", flush=True)
    items = []
    for r in para_results:
        items.append((f"orig__{r['doc_id']}", r["orig_text"]))
        items.append((f"para__{r['doc_id']}", r["para_text"]))

    # Bulk in Chunks von 50
    pangram_results = {}
    chunk = 50
    async with PangramClient() as pc:
        for i in range(0, len(items), chunk):
            sub = items[i:i+chunk]
            res = await pangram_batch(pc, sub, cache)
            pangram_results.update(res)
            save_cache(cache)  # incremental save
            print(f"  pangram {min(i+chunk, len(items))}/{len(items)}", flush=True)

    # Pairs zusammenstellen
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in para_results:
            orig_score = pangram_results.get(f"orig__{r['doc_id']}", {})
            para_score = pangram_results.get(f"para__{r['doc_id']}", {})
            pairs.append({
                "doc_id": r["doc_id"],
                "datum": r["datum"],
                "quelle": r["quelle"],
                "orig_text": r["orig_text"],
                "orig_fraction_ai": orig_score.get("fraction_ai"),
                "para_text": r["para_text"],
                "para_fraction_ai": para_score.get("fraction_ai"),
                "para_prediction": para_score.get("prediction"),
                "sonnet_cost_usd": r["sonnet_cost"],
            })
            f.write(json.dumps(pairs[-1], ensure_ascii=False) + "\n")

    n_orig_ok = sum(1 for p in pairs if p["orig_fraction_ai"] is not None)
    n_para_ok = sum(1 for p in pairs if p["para_fraction_ai"] is not None)
    n_para_below = sum(1 for p in pairs if (p["para_fraction_ai"] or 1.0) < 0.5)
    print(f"\n=== FERTIG ===")
    print(f"  Pairs:                {len(pairs)}")
    print(f"  Orig-Scores OK:       {n_orig_ok}")
    print(f"  Para-Scores OK:       {n_para_ok}")
    print(f"  Para fraction_ai < 0.5: {n_para_below}")
    print(f"  Sonnet-Kosten:        ${total_sonnet_cost:.2f}")
    print(f"  Output:               {OUT}")
    print(f"  Cache-Eintraege:      {len(cache)}")


if __name__ == "__main__":
    asyncio.run(main())
