"""Wackel-Test pangram_pre: 12 Phase-3a Casdorff-Originale + 1 Meinungsbeitrag (target).

ADR-008 §4 Pflicht-Vorlauf. Misst pangram_pre auf die Originaltexte,
um Fakes wie TSP__22f1d55f3 zu erkennen (war in Phase 2 Wackler trotz
pool-fraction_ai=1.0). Pre-Check für den 7000-char Demo-Text liefert den
Schwierigkeitsgrad für den Über-Nacht-Run.

Cost: 13 Pangram-Calls × $0.05 = $0.65
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()

from humanizer._pangram import PangramClient  # noqa: E402

EVAL = ROOT / "data" / "phase2-training-pool" / "eval.jsonl"
SAMPLE = ROOT / "data" / "phase3a" / "sample_n12.jsonl"
TARGET_TXT = ROOT / "data" / "test-corpora" / "target-text-2026-06-18-journalismus-tot.txt"
OUT_DIR = ROOT / "data" / "phase3b"
OUT = OUT_DIR / "pangram_pre_n13.jsonl"


async def main():
    sample_ids = [json.loads(l)["doc_id"] for l in SAMPLE.read_text().splitlines() if l.strip()]
    docs_all = {json.loads(l)["doc_id"]: json.loads(l)
                for l in EVAL.read_text().splitlines() if l.strip()}
    items = []
    for did in sample_ids:
        d = docs_all.get(did)
        if not d:
            print(f"WARN: {did} not in eval.jsonl", flush=True)
            continue
        items.append({"id": did, "text": d["volltext"], "chars": len(d["volltext"]),
                      "kind": "casdorff_phase3a"})
    items.append({"id": "TARGET_journalismus_tot_2026-06-18",
                  "text": TARGET_TXT.read_text(),
                  "chars": TARGET_TXT.stat().st_size,
                  "kind": "demo_target_ultra_long"})

    print(f"=== Pangram-Pre: {len(items)} Texte ===", flush=True)
    for it in items:
        print(f"  {it['id'][:36]:<36} {it['chars']:>5} chars  [{it['kind']}]", flush=True)

    async with PangramClient() as pc:
        res = await pc.check_bulk([{"id": it["id"], "text": it["text"]} for it in items])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        r = res.get(it["id"])
        row = {
            "doc_id": it["id"],
            "chars": it["chars"],
            "kind": it["kind"],
            "pangram_pre_fraction_ai": r.fraction_ai if r else None,
            "pangram_pre_fraction_ai_assisted": r.fraction_ai_assisted if r else None,
            "pangram_pre_fraction_human": r.fraction_human if r else None,
            "pangram_pre_prediction": r.prediction if r else "",
            "error": (r.error if r else "no-result"),
        }
        rows.append(row)

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    print(f"\n=== Resultate ===", flush=True)
    print(f"{'doc_id':<36} {'chars':>5} {'frac_ai':>8} {'pred':>10}  kind", flush=True)
    wackler = []
    for r in rows:
        fa = r["pangram_pre_fraction_ai"]
        fa_s = f"{fa:.3f}" if isinstance(fa, (int, float)) else "ERROR"
        pred = r["pangram_pre_prediction"][:10]
        print(f"{r['doc_id'][:36]:<36} {r['chars']:>5} {fa_s:>8} {pred:>10}  {r['kind']}", flush=True)
        if isinstance(fa, (int, float)) and fa < 0.5 and r["kind"] == "casdorff_phase3a":
            wackler.append(r["doc_id"])

    print(f"\nOutput: {OUT}", flush=True)
    if wackler:
        print(f"\n⚠ WACKLER (Casdorff-Originale mit pre_score<0.5): {wackler}", flush=True)
        print("  → Diese Docs aus Best-of-50-Sample werfen, sonst fake Bypass-Treffer.", flush=True)
    else:
        print("\n✓ Alle Casdorff-Originale pre_score>=0.5 — Sample sauber.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
