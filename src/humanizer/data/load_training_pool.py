"""Daten-Loader fuer Phase 2 Training-Pool.

Laedt NUR Artikel aus ki-check.db die Pangram mit fraction_ai = 1.0
(maximale Detection-Sicherheit) eingestuft hat. Stratifiziert nach Outlet,
deterministischer Split fuer Reproduzierbarkeit.

User-Direktive 2026-06-15: Training nur auf vollstaendig als KI klassifizierten Texten.
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

DB_DEFAULT = Path.home() / "Projects" / "ki-check" / "data" / "ki-check.db"

SEED = 2026_06_15


def load_pool(
    db_path: Path = DB_DEFAULT,
    min_chars: int = 500,
    require_fraction_ai: float = 1.0,
) -> list[dict]:
    """Vollstaendigen Trainings-Pool laden.

    Filter:
      - fraction_ai >= require_fraction_ai (default 1.0 → 100% AI-flagged)
      - Volltext vorhanden + Laenge > min_chars
      - dedupliziert nach doc_id
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT a.doc_id, a.quelle, a.doc_typ, a.datum, a.titel, a.autor,
               a.ressort, a.woerter, a.vorspann, a.volltext,
               c.fraction_ai, c.fraction_ai_assisted, c.fraction_human, c.prediction
        FROM article a
        JOIN check_result c ON c.doc_id = a.doc_id
        WHERE c.fraction_ai >= ?
          AND a.volltext IS NOT NULL
          AND LENGTH(a.volltext) > ?
        GROUP BY a.doc_id
        ORDER BY a.datum DESC
        """,
        (require_fraction_ai, min_chars),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def stratified_split(
    pool: list[dict],
    n_proxy: int = 180,
    n_train: int = 30,
    n_eval: int = 24,
    seed: int = SEED,
) -> dict[str, list[dict]]:
    """Pool stratifiziert nach Outlet auf {proxy, train, eval} aufteilen.

    Eval-Set ist EXKLUSIV — taucht nicht in Proxy/Train auf.
    """
    rng = random.Random(seed)

    by_outlet: dict[str, list[dict]] = {}
    for r in pool:
        by_outlet.setdefault(r["quelle"], []).append(r)
    for outlet in by_outlet.values():
        rng.shuffle(outlet)

    total_target = n_proxy + n_train + n_eval
    available = sum(len(v) for v in by_outlet.values())
    assert available >= total_target, (
        f"Pool zu klein: {available} verfuegbar, {total_target} angefordert"
    )

    proxy: list[dict] = []
    train: list[dict] = []
    eval_set: list[dict] = []

    # Proportional pro Outlet aufteilen
    for outlet, items in by_outlet.items():
        share = len(items) / available
        n_p = round(n_proxy * share)
        n_t = round(n_train * share)
        n_e = round(n_eval * share)
        proxy.extend(items[:n_p])
        train.extend(items[n_p:n_p + n_t])
        eval_set.extend(items[n_p + n_t:n_p + n_t + n_e])

    # Fill rest aus den groessten Pools (TSP/HBON), falls Runden gefehlt hat
    leftover = [r for outlet_items in by_outlet.values() for r in outlet_items
                if r["doc_id"] not in {x["doc_id"] for x in proxy + train + eval_set}]
    rng.shuffle(leftover)
    while len(proxy) < n_proxy and leftover:
        proxy.append(leftover.pop())
    while len(train) < n_train and leftover:
        train.append(leftover.pop())
    while len(eval_set) < n_eval and leftover:
        eval_set.append(leftover.pop())

    return {"proxy": proxy[:n_proxy], "train": train[:n_train], "eval": eval_set[:n_eval]}


def save_split(split: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in split.items():
        fp = out_dir / f"{name}.jsonl"
        with fp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"  {name}: {len(rows)} Artikel -> {fp}")


if __name__ == "__main__":
    pool = load_pool()
    print(f"Pool (fraction_ai = 1.0, Volltext > 500 chars): {len(pool)} Artikel")

    from collections import Counter
    outlets = Counter(r["quelle"] for r in pool)
    print(f"Outlet-Verteilung: {dict(outlets.most_common())}")

    split = stratified_split(pool, n_proxy=180, n_train=30, n_eval=24)
    print(f"\nSplit: proxy={len(split['proxy'])}, train={len(split['train'])}, eval={len(split['eval'])}")

    for name, rows in split.items():
        out_counter = Counter(r["quelle"] for r in rows)
        print(f"  {name}: {dict(out_counter.most_common())}")

    out_dir = Path(__file__).resolve().parents[3] / "data" / "phase2-training-pool"
    save_split(split, out_dir)
    print(f"\nGespeichert unter {out_dir}")
