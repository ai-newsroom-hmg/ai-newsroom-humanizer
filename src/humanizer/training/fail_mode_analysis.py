"""Phase-2-Fail-Mode-Diagnose — reproduzierbares Skript zu ADR 007.

Beantwortet drei Fragen:
  1. Was tut der Proxy-Loop wirklich? (Mean-Trajektorie pro Iter)
  2. Korrelieren Proxy-Scores mit Pangram-Scores? (Cross-Tabelle)
  3. Sind die 5 Erfolge ein Längen-Konfounder? (Längen-Verteilung pro Gruppe)

Input: data/phase2/eval_results.jsonl + loop_results.jsonl
Output: stdout-Tabelle, kein File-Write (Reports kommen aus ADR + git).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "data" / "phase2" / "eval_results.jsonl"
LOOP = ROOT / "data" / "phase2" / "loop_results.jsonl"


def main() -> None:
    ev = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    loop = {
        r["doc_id"]: r
        for r in (json.loads(l) for l in LOOP.read_text(encoding="utf-8").splitlines() if l.strip())
    }

    print(f"=== Phase-2 Fail-Mode-Diagnose (ADR 007) ===")
    print(f"n_eval={len(ev)} n_loop={len(loop)}")

    pre_human = [r for r in ev if r["pangram_pre"] < 1.0]
    print(f"\nLeaks (pangram_pre < 1.0): {len(pre_human)}")
    for r in pre_human:
        print(
            f"  {r['doc_id'][:14]} pre={r['pangram_pre']} "
            f"post={r['pangram_post']} bypass={r['bypass_success']}"
        )

    ok = [r for r in ev if r["bypass_success"] and r["pangram_pre"] == 1.0]
    fail = [r for r in ev if not r["bypass_success"]]

    def stats(rows: list[dict], key: str) -> str:
        vals = [r[key] for r in rows]
        return f"mean={st.mean(vals):.0f} median={st.median(vals):.0f} min={min(vals)} max={max(vals)}"

    print(f"\nLängen-Verteilung (Konfounder-Check):")
    print(f"  Bypass (n={len(ok)}): {stats(ok, 'orig_chars')}")
    print(f"  Fail   (n={len(fail)}): {stats(fail, 'orig_chars')}")

    print(f"\nProxy-Trajektorie (Loop-Wirkung):")
    traj = [
        [h.get("proxy_score") for h in r["history"] if "proxy_score" in h]
        for r in loop.values()
    ]
    max_iter = max(len(t) for t in traj)
    for it in range(max_iter):
        vals = [t[it] for t in traj if len(t) > it]
        print(f"  iter {it}: mean={st.mean(vals):.4f} std={st.stdev(vals):.4f} n={len(vals)}")

    deltas = [r["proxy_post"] - r["proxy_pre"] for r in ev]
    print(f"\nΔ proxy_post − proxy_pre über alle {len(ev)}:")
    print(
        f"  mean={st.mean(deltas):+.4f} std={st.stdev(deltas):.4f} "
        f"min={min(deltas):+.4f} max={max(deltas):+.4f}"
    )

    print(f"\nProxy-Post vs Pangram-Post bei den {len(ok)} Bypass-Erfolgen:")
    for r in ok:
        print(
            f"  {r['doc_id'][:14]} proxy_post={r['proxy_post']:.3f} "
            f"pangram_post={r['pangram_post']:.3f} chars={r['orig_chars']}"
        )

    print(f"\nBereinigte Bypass-Rate: {len(ok)}/{len(ev) - len(pre_human)} "
          f"= {len(ok) / (len(ev) - len(pre_human)) * 100:.0f} % "
          f"(nach Leak-Bereinigung)")


if __name__ == "__main__":
    main()
