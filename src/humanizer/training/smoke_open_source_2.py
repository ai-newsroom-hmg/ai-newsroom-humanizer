"""Phase-3-Smoke-2: Mistral-Family auf 3 long-form Fails — Reproduktion + Generalisierung.

Smoke-1 zeigte: Mistral-Small-3.2 @ temp=1.1 hat HB_100214957 von pangram=1.0 auf
0.510 (Mixed, BGE=0.91 faithful) bewegt — Sonnet schaffte das nie. Frage jetzt:

  1. Ist das Signal reproduzierbar auf anderen Artikeln?
  2. Hilft ein groesseres Mistral (Large-2512) oder das kleine 24B besser?
  3. Wie verteilt sich das Signal ueber Temperaturen?

Setup: 3 Artikel × {Mistral-Small-3.2-24b, Mistral-Large-2512} × 10 Varianten @ Temp 0.8-1.3
Cost-Budget: ~$2.50 (60 gen ~$0.50 + 60 pangram ~$3 -- aber viele Cache-Hits erwartet).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanizer._openrouter import ORClient  # noqa: E402

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
from humanizer._pangram import PangramClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
LOOP = ROOT / "data" / "phase2" / "loop_results.jsonl"
CACHE = ROOT / "data" / "phase2" / "pangram_cache.json"
OUT_DIR = ROOT / "data" / "phase3-smoke"
OUT = OUT_DIR / "open_source_smoke2.jsonl"

# Drei long-form fails aus Phase 2 — verschiedene Quellen
TARGETS = ["HB_100214957", "TSP__074e1b8da3...", "WWON__45d50bf9..."]
# Match by prefix in loop_results
TARGET_PREFIXES = ["HB_100214957", "TSP__074e1b8d", "WWON__45d50bf"]

MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
    "mistralai/mistral-large-2512",
]

TEMPS = [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]

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


async def gen_variant(client: ORClient, text: str, temp: float) -> dict:
    try:
        out = await client.complete(SYSTEM_PROMPT, USER.format(text=text),
                                    temperature=temp, max_tokens=3000)
        return {"text": out["text"].strip(), "cost": out["cost_usd"], "temp": temp, "error": None}
    except Exception as e:
        return {"text": "", "cost": 0.0, "temp": temp, "error": str(e)[:200]}


async def main():
    arts_all = [json.loads(l) for l in LOOP.read_text(encoding="utf-8").splitlines() if l.strip()]
    arts = []
    for prefix in TARGET_PREFIXES:
        match = next((a for a in arts_all if a["doc_id"].startswith(prefix)), None)
        if match:
            arts.append(match)
        else:
            print(f"WARN: no match for prefix {prefix}", flush=True)
    print(f"=== Smoke-2: {len(arts)} Artikel × {len(MODELS)} Modelle × {len(TEMPS)} Temperaturen ===",
          flush=True)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"    Cache: {len(cache)} Eintraege", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants_all: list[dict] = []

    for art in arts:
        did = art["doc_id"]
        orig = art["orig_text"]
        print(f"\n=== {did} ({len(orig)} chars) ===", flush=True)
        for model in MODELS:
            print(f"  --- {model} ---", flush=True)
            client = ORClient(model=model)
            t0 = time.time()
            results = await asyncio.gather(*[gen_variant(client, orig, t) for t in TEMPS])
            gen_cost = sum(r["cost"] for r in results)
            n_ok = sum(1 for r in results if not r["error"])
            print(f"      {n_ok}/{len(TEMPS)} gen in {time.time()-t0:.0f}s, ${gen_cost:.4f}",
                  flush=True)
            for i, r in enumerate(results):
                if r["error"]:
                    continue
                variants_all.append({
                    "doc_id": did,
                    "orig_chars": len(orig),
                    "model": model,
                    "temp": r["temp"],
                    "idx": i,
                    "text": r["text"],
                    "gen_cost": r["cost"],
                    "chars": len(r["text"]),
                })

    # Pangram
    items = []
    for v in variants_all:
        vid = (f"{v['doc_id']}__{v['model'].split('/')[-1]}__t{v['temp']}__i{v['idx']}")
        v["vid"] = vid
        items.append((vid, v["text"]))

    print(f"\n--- Pangram ({len(items)} Texte, Cache-aware) ---", flush=True)
    api_items = []
    pangram_results: dict[str, dict] = {}
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

    with OUT.open("w", encoding="utf-8") as f:
        for v in variants_all:
            p = pangram_results.get(v["vid"], {})
            v["pangram_fraction_ai"] = p.get("fraction_ai")
            v["pangram_prediction"] = p.get("prediction")
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    # Bericht
    print("\n=== Per-Doc × Modell ===", flush=True)
    print(f"{'doc_id':<24} {'model':<32} {'n':>3} {'min':>6} {'mean':>6} {'<0.5':>5} {'<0.2':>5}",
          flush=True)
    for art in arts:
        did = art["doc_id"][:24]
        for model in MODELS:
            rows = [v for v in variants_all
                    if v["doc_id"] == art["doc_id"] and v["model"] == model
                    and v.get("pangram_fraction_ai") is not None]
            if not rows:
                continue
            fs = [v["pangram_fraction_ai"] for v in rows]
            print(f"{did:<24} {model.split('/')[-1]:<32} {len(rows):>3} "
                  f"{min(fs):>6.3f} {mean(fs):>6.3f} "
                  f"{sum(1 for x in fs if x<0.5):>5} {sum(1 for x in fs if x<0.2):>5}", flush=True)

    print("\n=== Bewegung-Verteilung über Temp ===", flush=True)
    by_temp: dict[float, list[float]] = {}
    for v in variants_all:
        if v.get("pangram_fraction_ai") is None:
            continue
        by_temp.setdefault(v["temp"], []).append(v["pangram_fraction_ai"])
    for t in sorted(by_temp):
        fs = by_temp[t]
        print(f"  temp={t:>5}: n={len(fs)} min={min(fs):.3f} mean={mean(fs):.3f} "
              f"max={max(fs):.3f} <0.5: {sum(1 for x in fs if x<0.5)}", flush=True)

    print("\n=== Top-5 Variants gesamt ===", flush=True)
    sorted_v = sorted([v for v in variants_all if v.get("pangram_fraction_ai") is not None],
                      key=lambda x: x["pangram_fraction_ai"])[:5]
    for v in sorted_v:
        print(f"  {v['doc_id'][:14]} {v['model'].split('/')[-1][:24]:<24} "
              f"temp={v['temp']:>5} fraction_ai={v['pangram_fraction_ai']:.3f} "
              f"pred={v.get('pangram_prediction','')}", flush=True)

    print(f"\nOutput: {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
