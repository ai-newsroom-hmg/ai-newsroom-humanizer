"""Phase-3-Smoke: Open-Source-Generatoren auf 1 long-form Phase-2-Fail.

Hypothese: deutsch-native / Apache-2.0-Modelle haben eine andere AI-Signatur als
Sonnet — Pangram (trainiert vermutlich schwerpunktmaessig auf GPT/Claude) koennte
sie weniger gut erkennen.

Vorgehen:
  1. Lade HB_100214957 (2.943 chars, Phase-2-Fail, pangram_pre=1.0)
  2. Generiere je Modell 10 Varianten (Temp-Sweep 0.6..1.1) parallel
  3. Pangram-scoren alle Varianten + Original (SHA-Cache nutzen)
  4. Output: Tabelle min/mean/max fraction_ai pro Modell

Cost-Budget: ~$1.20 (gen ~$0.20, 20 pangram-calls ~$1).
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
OUT = OUT_DIR / "open_source_variants.jsonl"

TARGET_DOC = "HB_100214957"

MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
]

TEMPS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]

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
    arts = [json.loads(l) for l in LOOP.read_text(encoding="utf-8").splitlines() if l.strip()]
    art = next((a for a in arts if a["doc_id"] == TARGET_DOC), None)
    if not art:
        sys.exit(f"Target {TARGET_DOC} nicht in {LOOP}")
    orig = art["orig_text"]
    print(f"=== Smoke: {TARGET_DOC} ({len(orig)} chars, "
          f"Phase-2 pangram_post = 1.0, proxy_post={art['proxy_score_post']:.3f}) ===", flush=True)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"    Cache: {len(cache)} Eintraege", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants_all: list[dict] = []
    for model in MODELS:
        print(f"\n--- {model} ---", flush=True)
        client = ORClient(model=model)
        t0 = time.time()
        results = await asyncio.gather(*[gen_variant(client, orig, t) for t in TEMPS])
        gen_cost = sum(r["cost"] for r in results)
        n_ok = sum(1 for r in results if not r["error"])
        print(f"    {n_ok}/{len(TEMPS)} Generierungen in {time.time()-t0:.0f}s, ${gen_cost:.4f}",
              flush=True)
        for i, r in enumerate(results):
            if r["error"]:
                print(f"    err temp={r['temp']}: {r['error']}", flush=True)
                continue
            variants_all.append({
                "model": model,
                "temp": r["temp"],
                "idx": i,
                "text": r["text"],
                "gen_cost": r["cost"],
                "chars": len(r["text"]),
            })

    if not variants_all:
        sys.exit("Keine erfolgreichen Generierungen.")

    items = [(f"orig__{TARGET_DOC}", orig)]
    for v in variants_all:
        vid = f"{v['model'].split('/')[-1]}__t{v['temp']}__i{v['idx']}"
        v["vid"] = vid
        items.append((vid, v["text"]))

    print(f"\n--- Pangram ({len(items)} Texte, Cache-aware) ---", flush=True)
    pangram_results: dict[str, dict] = {}
    api_items = []
    for tid, txt in items:
        h = sha(txt)
        if h in cache:
            pangram_results[tid] = cache[h]
        else:
            api_items.append({"id": tid, "text": txt, "_sha": h})
    print(f"    Cache-Hits: {len(items)-len(api_items)}/{len(items)}", flush=True)

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

    orig_pangram = pangram_results.get(f"orig__{TARGET_DOC}", {}).get("fraction_ai")
    print(f"    Orig pangram_fraction_ai = {orig_pangram}", flush=True)

    with OUT.open("w", encoding="utf-8") as f:
        for v in variants_all:
            p = pangram_results.get(v["vid"], {})
            v["pangram_fraction_ai"] = p.get("fraction_ai")
            v["pangram_prediction"] = p.get("prediction")
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"\n=== Per-Modell-Verteilung ===", flush=True)
    print(f"{'Modell':<48} {'n':>3} {'min':>6} {'mean':>6} {'max':>6} "
          f"{'<0.5':>5} {'<0.2':>5}", flush=True)
    for model in MODELS:
        rows = [v for v in variants_all if v["model"] == model and v.get("pangram_fraction_ai") is not None]
        if not rows:
            print(f"{model:<48} keine gueltigen Ergebnisse", flush=True)
            continue
        fs = [v["pangram_fraction_ai"] for v in rows]
        n_below_5 = sum(1 for x in fs if x < 0.5)
        n_below_2 = sum(1 for x in fs if x < 0.2)
        print(f"{model:<48} {len(rows):>3} {min(fs):>6.3f} {mean(fs):>6.3f} {max(fs):>6.3f} "
              f"{n_below_5:>5} {n_below_2:>5}", flush=True)

    print(f"\n=== Top-3 Variants pro Modell (sortiert nach fraction_ai aufsteigend) ===",
          flush=True)
    for model in MODELS:
        rows = sorted(
            [v for v in variants_all if v["model"] == model and v.get("pangram_fraction_ai") is not None],
            key=lambda x: x["pangram_fraction_ai"],
        )[:3]
        for v in rows:
            print(f"  [{model.split('/')[-1]}] temp={v['temp']} "
                  f"fraction_ai={v['pangram_fraction_ai']:.3f} "
                  f"pred={v.get('pangram_prediction','')} chars={v['chars']}", flush=True)

    print(f"\nOutput: {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
