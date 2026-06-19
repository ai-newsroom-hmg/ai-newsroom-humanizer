"""Pangram FPR-Eval auf pre-2022-Korpus.

Lädt data/fpr-test/pre2022_corpus.jsonl, ruft Pangram-Bulk-API, schreibt
data/fpr-test/pangram_results.jsonl + console-Report.

Cost: ~$0.05 pro Doc × 100 = $5.

Pangram-claimed FPR auf pre-ChatGPT human text (Tech Report 2024): ≤ 0.5 %.
UChicago Booth 2026 (n=1992): ≤ 1 % FPR auf pre-2020.
Frage: gilt das auch für deutsche Leitmedien?
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()

from humanizer._pangram import PangramClient  # noqa: E402

CORPUS = ROOT / "data" / "fpr-test" / "pre2022_corpus.jsonl"
OUT = ROOT / "data" / "fpr-test" / "pangram_results.jsonl"


def categorize(fraction_ai: float) -> str:
    if fraction_ai >= 0.8:
        return "AI"
    if fraction_ai >= 0.5:
        return "Mixed-AI"
    if fraction_ai >= 0.2:
        return "Mixed-Human"
    return "Human"


async def main():
    docs = [json.loads(l) for l in CORPUS.read_text().split("\n") if l.strip()]
    print(f"=== Pangram FPR-Eval auf {len(docs)} pre-2022 Texten ===", flush=True)

    items = [{"id": d["doc_id"], "text": d["text"]} for d in docs]
    print(f"  Cost-Estimate: ${len(items) * 0.05:.2f}", flush=True)
    print(f"  Pre-Check Sample-Source 1: {docs[0]['source']} {docs[0]['pub_date']} "
          f"{docs[0]['chars']}c", flush=True)

    async with PangramClient() as pc:
        res = await pc.check_bulk(items)

    rows = []
    for d in docs:
        r = res.get(d["doc_id"])
        row = {
            "doc_id": d["doc_id"], "source": d["source"], "pub_date": d["pub_date"],
            "chars": d["chars"], "headline": d.get("headline", ""),
            "ressort": d.get("ressort", ""),
            "fraction_ai": r.fraction_ai if r and r.fraction_ai is not None else None,
            "fraction_ai_assisted": r.fraction_ai_assisted if r else None,
            "fraction_human": r.fraction_human if r else None,
            "prediction": r.prediction if r else "",
            "error": (r.error if r else "no-result") if not (r and r.fraction_ai is not None) else "",
        }
        rows.append(row)

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"\nOutput: {OUT}", flush=True)

    # Aggregate
    ok = [r for r in rows if r["fraction_ai"] is not None]
    err = [r for r in rows if r["fraction_ai"] is None]
    print(f"\n=== FPR-Analyse n={len(ok)} (err={len(err)}) ===", flush=True)

    # Overall buckets
    cats = Counter(categorize(r["fraction_ai"]) for r in ok)
    print(f"\nPangram-Kategorien:", flush=True)
    for cat in ("Human", "Mixed-Human", "Mixed-AI", "AI"):
        n = cats.get(cat, 0)
        print(f"  {cat:<13} {n:>3}/{len(ok)} = {n / len(ok) * 100:.1f} %", flush=True)

    # False-Positive Definitions
    fp_strict = sum(1 for r in ok if r["fraction_ai"] >= 0.5)   # "AI" or "Mixed-AI"
    fp_loose = sum(1 for r in ok if r["fraction_ai"] >= 0.2)    # "Mixed-Human" upward
    fp_hard = sum(1 for r in ok if r["fraction_ai"] >= 0.8)     # strict "AI"
    print(f"\nFalse-Positive-Raten:", flush=True)
    print(f"  Hard FP (fraction_ai >= 0.8):  {fp_hard}/{len(ok)} = {fp_hard / len(ok) * 100:.1f} %", flush=True)
    print(f"  Strict FP (>= 0.5):            {fp_strict}/{len(ok)} = {fp_strict / len(ok) * 100:.1f} %", flush=True)
    print(f"  Loose FP (>= 0.2):             {fp_loose}/{len(ok)} = {fp_loose / len(ok) * 100:.1f} %", flush=True)
    print(f"\n  Pangram claimed (Tech Report 2024): ≤ 0.5 %", flush=True)
    print(f"  UChicago Booth 2026 (n=1992):       ≤ 1 %", flush=True)

    # Per Jahr
    by_year = {}
    for r in ok:
        y = r["pub_date"][:4]
        by_year.setdefault(y, []).append(r)
    print(f"\nPer Jahr:", flush=True)
    for y in sorted(by_year):
        rs = by_year[y]
        fp = sum(1 for r in rs if r["fraction_ai"] >= 0.5)
        print(f"  {y}: FP {fp}/{len(rs)} = {fp / max(len(rs), 1) * 100:.1f} %", flush=True)

    # Per Source
    by_src = {}
    for r in ok:
        by_src.setdefault(r["source"], []).append(r)
    print(f"\nPer Quelle (Top 10 nach n):", flush=True)
    src_sorted = sorted(by_src.items(), key=lambda kv: -len(kv[1]))[:10]
    for src, rs in src_sorted:
        fp = sum(1 for r in rs if r["fraction_ai"] >= 0.5)
        print(f"  {src[:35]:<35} n={len(rs):>3}  FP {fp}/{len(rs)} = {fp / len(rs) * 100:.1f} %", flush=True)

    # Per Länge
    print(f"\nPer Länge:", flush=True)
    buckets = {"short (<1500)": [], "mid (1500-3000)": [], "long (>=3000)": []}
    for r in ok:
        if r["chars"] < 1500: buckets["short (<1500)"].append(r)
        elif r["chars"] < 3000: buckets["mid (1500-3000)"].append(r)
        else: buckets["long (>=3000)"].append(r)
    for k, rs in buckets.items():
        if not rs: continue
        fp = sum(1 for r in rs if r["fraction_ai"] >= 0.5)
        print(f"  {k:<20} n={len(rs):>3}  FP {fp}/{len(rs)} = {fp / len(rs) * 100:.1f} %", flush=True)

    # Worst offenders
    print(f"\nTop 5 verdächtigste pre-2022 Texte (höchste fraction_ai):", flush=True)
    worst = sorted(ok, key=lambda r: -r["fraction_ai"])[:5]
    for r in worst:
        print(f"  fraction_ai={r['fraction_ai']:.3f} {r['source'][:25]:<25} {r['pub_date']} "
              f"{r['chars']:>5}c — {r['headline'][:60]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
