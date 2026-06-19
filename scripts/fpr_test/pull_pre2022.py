"""Pull 100 random pre-ChatGPT (2019-01-01 bis 2021-12-31) Texte aus
deutschen Tier-1-Leitmedien via Genios — für Pangram FPR-Test.

Strategie:
1. 100 random Tage aus 2019-2021 sample.
2. Pro Tag: search_filtered_page mit allen TIER1_PRINT_NAMES,
   1 random Document picken, get_document → volltext.
3. Skip wenn < 800 chars oder Search-Page leer.

Output: data/fpr-test/pre2022_corpus.jsonl mit Felder:
   doc_id, source, headline, pub_date, fields (Genios raw), text, chars

Ausführen MIT signals-identifier-print venv:
   ~/Projects/signals-identifier-print/.venv/bin/python scripts/fpr_test/pull_pre2022.py

Cost: 0 (HMG Genios-Lizenz, internal).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

# Cross-repo import via venv that has signals_identifier on sys.path
SI_PRINT_SRC = Path.home() / "Projects" / "signals-identifier-print" / "src"
if str(SI_PRINT_SRC) not in sys.path:
    sys.path.insert(0, str(SI_PRINT_SRC))

from signals_identifier.pressearchiv import PressearchivClient, TIER1_PRINT_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "fpr-test"
OUT_JSONL = OUT_DIR / "pre2022_corpus.jsonl"
TARGET_N = 100
MIN_CHARS = 800
START = date(2019, 1, 1)
END = date(2021, 12, 31)


def iso_to_de(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def random_dates(n: int, seed: int = 42) -> list[date]:
    rng = random.Random(seed)
    delta = (END - START).days
    days = rng.sample(range(delta + 1), min(n * 3, delta + 1))  # over-sample
    return [START + timedelta(days=x) for x in days]


async def fetch_random_doc_for_day(client: PressearchivClient, day: date,
                                   rng: random.Random) -> dict | None:
    """Search day across all TIER1 sources, pick a random doc, fetch volltext."""
    de = iso_to_de(day)
    try:
        page = await client.search_filtered_page(
            "", sources=TIER1_PRINT_NAMES,
            date_from=de, date_to=de,
            offset=0, size=100,
            get_sources=False,
        )
    except Exception as e:
        print(f"  [{day}] search failed: {str(e)[:120]}", flush=True)
        return None
    docs = page.get("documents", []) if isinstance(page, dict) else []
    if not docs:
        return None
    rng.shuffle(docs)
    for d in docs[:10]:  # try up to 10 random picks
        doc_id = d.get("documentId")
        if not doc_id:
            continue
        try:
            doc = await client.get_document(doc_id)
        except Exception as e:
            print(f"  [{day}] {doc_id[:24]} fetch failed: {str(e)[:120]}", flush=True)
            continue
        text = (doc.clean_text or "").strip()
        if len(text) < MIN_CHARS:
            continue
        fields = d.get("fields", {}) if isinstance(d, dict) else {}
        return {
            "doc_id": doc_id,
            "source": fields.get("QUELLENNAME") or d.get("dbShortcut") or doc.source,
            "headline": fields.get("TL-TITEL") or fields.get("ARTIKELTITEL") or doc.title,
            "pub_date": day.isoformat(),
            "ressort": fields.get("RESSORT") or "",
            "autor": fields.get("AUTOR") or "",
            "chars": len(text),
            "text": text,
        }
    return None


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    candidate_days = random_dates(TARGET_N, seed=42)
    print(f"=== FPR-Korpus Pull pre-2022 ===", flush=True)
    print(f"  Target: {TARGET_N} Texte, MIN_CHARS={MIN_CHARS}", flush=True)
    print(f"  Range: {START} bis {END} ({(END - START).days + 1} Tage)", flush=True)
    print(f"  Sources: {len(TIER1_PRINT_NAMES)} TIER1 print + online", flush=True)
    print(f"  Candidate days sampled: {len(candidate_days)}", flush=True)
    print(f"  Output: {OUT_JSONL}", flush=True)

    collected: list[dict] = []
    seen_ids: set[str] = set()
    async with PressearchivClient() as client:
        await client.login()
        for i, day in enumerate(candidate_days, 1):
            if len(collected) >= TARGET_N:
                break
            doc = await fetch_random_doc_for_day(client, day, rng)
            if not doc:
                continue
            if doc["doc_id"] in seen_ids:
                continue
            seen_ids.add(doc["doc_id"])
            collected.append(doc)
            print(f"  [{len(collected):3d}/{TARGET_N}] {day} {doc['source'][:25]:<25} "
                  f"{doc['chars']:>5}c {(doc['headline'] or '')[:60]}", flush=True)
            # write progressively
            with OUT_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n=== Pulled {len(collected)} docs → {OUT_JSONL} ===", flush=True)
    # Stats
    from collections import Counter
    by_year = Counter(d["pub_date"][:4] for d in collected)
    by_source = Counter(d["source"] for d in collected)
    by_len_bucket = Counter()
    for d in collected:
        if d["chars"] < 1500: by_len_bucket["short(<1500)"] += 1
        elif d["chars"] < 3000: by_len_bucket["mid(1500-3000)"] += 1
        else: by_len_bucket["long(>=3000)"] += 1
    print(f"\nPer Jahr: {dict(by_year)}", flush=True)
    print(f"Per Quelle (top10): {dict(by_source.most_common(10))}", flush=True)
    print(f"Per Länge: {dict(by_len_bucket)}", flush=True)


if __name__ == "__main__":
    # Truncate output before run
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()
    asyncio.run(main())
