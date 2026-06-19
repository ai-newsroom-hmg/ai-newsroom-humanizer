"""Phase-3a Mini-Replikation: n=12 length-stratified Docs, nur Mistral-Small-3.2-24b.

Smoke-2 (n=3) zeigte: alle 3 Docs hatten je >=1 inhaltstreue Bypass-Variante,
Sweet-Spot Mistral-Small-3.2 @ temp 1.0-1.1. Diese Phase prueft, ob das auf
groesserem Sample reproduziert — gezielt aus eval.jsonl, ohne die Smoke-2-Docs.

Setup: 12 Artikel (4 short / 3 mid / 5 long, fraction_ai=1.0) × Mistral-Small-3.2-24b
       × 6 Temps (0.9, 0.95, 1.0, 1.05, 1.10, 1.15) × 4 idx = 24 Varianten/Doc.

Budget: 288 OpenRouter-Calls (~$0.10) + 288 Pangram-Calls ($14.40 nominal, viel
Cache erwartet — Smoke-2 hatte Cache von 60). Hartes Budget-Limit unten.
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
from humanizer._openrouter import ORClient  # noqa: E402

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
from humanizer._pangram import PangramClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "data" / "phase2-training-pool" / "eval.jsonl"
CACHE = ROOT / "data" / "phase2" / "pangram_cache.json"
OUT_DIR = ROOT / "data" / "phase3a"
OUT = OUT_DIR / "results_n12.jsonl"
SAMPLE_OUT = OUT_DIR / "sample_n12.jsonl"

# Smoke-2-Docs ausschliessen (Reproduktion auf NEUEN Docs)
EXCLUDE_PREFIXES = ("HB_100214957", "TSP__074e1b8d", "WWON__45d50bf9")

MODEL = "mistralai/mistral-small-3.2-24b-instruct"
TEMPS = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
N_IDX = 4
MAX_PANGRAM_CALLS = 320  # hard cap → $16

SYSTEM_PROMPT = """Du bist ein erfahrener deutscher Journalist. Schreibe den
folgenden Text so um, dass er menschlich klingt — variable Satzlaengen,
gelegentliche Brueche im Rhythmus, idiomatische Wendungen, eigene Wortwahl.
Inhalt, Fakten, Namen, Zahlen, Zitate bleiben EXAKT erhalten.

Vermeide:
- Floskel-Anschluesse ('daher', 'in diesem Kontext', 'vor diesem Hintergrund')
- Erstens / Zweitens / Drittens-Strukturen
- Drei parallele Adjektive ('klar, transparent, nachvollziehbar')
- Glatte Aufzaehlungen

Antworte NUR mit dem umgeschriebenen Text. Keine Vorrede."""

USER = "Originaltext:\n\n{text}\n\nMenschlich umgeschrieben:"


def sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def pick_sample(docs: list[dict]) -> list[dict]:
    """4 short + 3 mid + 5 long aus eligible Pool (fraction_ai=1.0, len>=500)."""
    elig = [d for d in docs if d.get("fraction_ai") == 1.0 and len(d.get("volltext", "")) >= 500
            and not d["doc_id"].startswith(EXCLUDE_PREFIXES)]
    short = sorted([d for d in elig if len(d["volltext"]) <= 1400], key=lambda x: len(x["volltext"]))
    mid = sorted([d for d in elig if 1400 < len(d["volltext"]) <= 2800], key=lambda x: len(x["volltext"]))
    lng = sorted([d for d in elig if len(d["volltext"]) > 2800], key=lambda x: len(x["volltext"]))
    print(f"[sample] pool: short={len(short)} mid={len(mid)} long={len(lng)}", flush=True)
    return short[:4] + mid[:3] + lng[:5]


async def gen_variant(client: ORClient, text: str, temp: float) -> dict:
    try:
        out = await client.complete(SYSTEM_PROMPT, USER.format(text=text),
                                    temperature=temp, max_tokens=3000)
        return {"text": out["text"].strip(), "cost": out["cost_usd"], "temp": temp, "error": None}
    except Exception as e:
        return {"text": "", "cost": 0.0, "temp": temp, "error": str(e)[:200]}


async def main():
    docs_all = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = pick_sample(docs_all)
    print(f"=== Sample n={len(sample)} ===", flush=True)
    for d in sample:
        print(f"  {d['doc_id'][:24]:<24} {len(d['volltext']):>5} chars  {(d.get('autor') or '?')[:30]}",
              flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUT.write_text("\n".join(
        json.dumps({"doc_id": d["doc_id"], "chars": len(d["volltext"]), "autor": d.get("autor")},
                   ensure_ascii=False) for d in sample), encoding="utf-8")

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"[pangram] cache size: {len(cache)}", flush=True)

    variants_all: list[dict] = []
    client = ORClient(model=MODEL)
    for d in sample:
        did = d["doc_id"]; orig = d["volltext"]
        print(f"\n=== {did[:30]:<30} ({len(orig)} chars) ===", flush=True)
        t0 = time.time()
        tasks = [gen_variant(client, orig, t) for t in TEMPS for _ in range(N_IDX)]
        results = await asyncio.gather(*tasks)
        gen_cost = sum(r["cost"] for r in results)
        n_ok = sum(1 for r in results if not r["error"])
        print(f"  gen: {n_ok}/{len(results)} in {time.time()-t0:.0f}s  ${gen_cost:.4f}", flush=True)
        i = 0
        for tmp in TEMPS:
            for k in range(N_IDX):
                r = results[i]; i += 1
                if r["error"]: continue
                variants_all.append({
                    "doc_id": did, "orig_chars": len(orig), "model": MODEL,
                    "temp": tmp, "idx": k, "text": r["text"], "gen_cost": r["cost"],
                    "chars": len(r["text"]),
                })

    items = []
    for v in variants_all:
        vid = f"{v['doc_id'][:24]}__t{v['temp']}__i{v['idx']}"
        v["vid"] = vid
        items.append((vid, v["text"], sha(v["text"])))

    cache_hits = sum(1 for _, _, h in items if h in cache)
    api_n = len(items) - cache_hits
    print(f"\n[pangram] {len(items)} texts: {cache_hits} cached, {api_n} api-calls → ${api_n*0.05:.2f}",
          flush=True)
    if api_n > MAX_PANGRAM_CALLS:
        print(f"  ABORT: {api_n} > {MAX_PANGRAM_CALLS} budget cap", flush=True)
        return

    pangram_results: dict[str, dict] = {}
    for vid, _, h in items:
        if h in cache:
            pangram_results[vid] = cache[h]
    api_items = [{"id": vid, "text": txt, "_sha": h}
                 for vid, txt, h in items if h not in cache]
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

    with OUT.open("w", encoding="utf-8") as f:
        for v in variants_all:
            p = pangram_results.get(v["vid"], {})
            v["pangram_fraction_ai"] = p.get("fraction_ai")
            v["pangram_prediction"] = p.get("prediction")
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"\n=== Per-Doc ({MODEL.split('/')[-1]}) ===", flush=True)
    print(f"{'doc_id':<26} {'chars':>5} {'n':>3} {'best_p':>7} {'<0.5':>5} {'<0.2':>5}", flush=True)
    by_doc: dict[str, list[dict]] = {}
    for v in variants_all:
        if v.get("pangram_fraction_ai") is not None:
            by_doc.setdefault(v["doc_id"], []).append(v)
    bypass_docs = 0
    for d in sample:
        rows = by_doc.get(d["doc_id"], [])
        if not rows: continue
        fs = [v["pangram_fraction_ai"] for v in rows]
        b02 = sum(1 for x in fs if x < 0.2)
        if b02: bypass_docs += 1
        print(f"{d['doc_id'][:26]:<26} {len(d['volltext']):>5} {len(rows):>3} "
              f"{min(fs):>7.3f} {sum(1 for x in fs if x<0.5):>5} {b02:>5}", flush=True)
    print(f"\n>>> Docs mit >=1 Bypass-Variante (P<0.2): {bypass_docs}/{len(sample)} = "
          f"{bypass_docs/len(sample)*100:.0f}%", flush=True)
    print(f"\nOutput: {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
