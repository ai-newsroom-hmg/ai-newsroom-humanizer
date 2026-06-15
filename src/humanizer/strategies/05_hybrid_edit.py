"""Humanize-Strategie 05 — Hybrid-Edit (Source-Confusion / Zindulka 2024).

Theorie: Pangram lernt 'LLM-Schreibsignatur'. Wenn wir echte menschliche Saetze
in den KI-Text injizieren, bricht die Signatur lokal. Das Source-Monitoring-
Konzept (Zindulka 2024): wer zwei Quellen mischt, verliert die KI-Spur.

Strategie:
  1. KI-Text in Saetze splitten
  2. Aus dem Casdorff-Human-Pool (28 Artikel der DB) Saetze pullen
  3. Alle 3-5 Saetze einen menschlichen Casdorff-Satz einbauen
  4. Pangram-Check

Plus Strategie 06 (Mikro-Adversarial) als zweite Schicht:
  - SpaceInfi: Whitespace vor Komma
  - Unicode-Homoglyph (Cyrillic e statt latin e, sparsam)
  - Article-Deletion (regulator)

Pilot: 5 Casdorff-AI-Artikel.
Output: data/humanize/05_hybrid_edit_results.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
sys.path.insert(0, str(Path.home() / "Projects" / "ki-check" / "src"))
from ki_check.pangram import PangramClient  # noqa: E402

ROOT = Path.home() / "Projects" / "ki-check"
DB = ROOT / "data" / "ki-check.db"
OUT = ROOT / "data" / "humanize" / "05_hybrid_edit_results.jsonl"

THRESHOLD = 0.20
LIMIT = int(os.environ.get("HUMANIZE_LIMIT", "5"))
HUMAN_INJECT_EVERY = 4   # alle 4 KI-Saetze einen Human-Satz einsetzen
SEED = 2026_06_15


# ── Casdorff-Datenbasis aus SQLite ─────────────────────────────────────────
def load_human_pool() -> list[str]:
    """Saetze aus Casdorff-Artikel die Pangram als Human eingestuft hat."""
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT a.volltext FROM article a
        JOIN check_result c ON c.doc_id=a.doc_id
        WHERE a.autor LIKE '%Casdorff%' AND c.prediction='Human'
          AND a.volltext IS NOT NULL
    """).fetchall()
    con.close()
    sentences = []
    for r in rows:
        sentences.extend(split_sentences(r["volltext"]))
    # Filter: nur Saetze die journalistisch greifbar sind (>40 chars, <300 chars)
    return [s for s in sentences if 40 <= len(s) <= 300]


def load_ai_articles(limit: int) -> list[dict]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT a.doc_id, a.datum, a.titel, a.autor, a.woerter, a.vorspann, a.volltext,
               c.fraction_ai AS pai_pre, c.fraction_ai_assisted AS paa_pre,
               c.fraction_human AS phu_pre, c.prediction AS pred_pre
        FROM article a
        JOIN check_result c ON c.doc_id=a.doc_id
        WHERE a.autor LIKE '%Casdorff%' AND c.prediction='AI'
          AND a.volltext IS NOT NULL AND LENGTH(a.volltext) > 800
        GROUP BY a.doc_id
        ORDER BY c.fraction_ai DESC, a.datum DESC
        LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── Satz-Splitting + Mischen ───────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def inject_human_sentences(ai_text: str, human_pool: list[str], rng: random.Random,
                            every: int = HUMAN_INJECT_EVERY) -> str:
    ai_sents = split_sentences(ai_text)
    out: list[str] = []
    for i, s in enumerate(ai_sents):
        out.append(s)
        if (i + 1) % every == 0:
            out.append(rng.choice(human_pool))
    return " ".join(out)


# ── Strategie 06: Mikro-Adversarial ─────────────────────────────────────────
HOMOGLYPHS = {
    "a": "а",  # Cyrillic 'а' (U+0430)
    "e": "е",  # Cyrillic 'е' (U+0435)
    "o": "о",  # Cyrillic 'о' (U+043E)
    "p": "р",  # Cyrillic 'р' (U+0440)
    "c": "с",  # Cyrillic 'с' (U+0441)
    "x": "х",  # Cyrillic 'х' (U+0445)
}


def micro_adversarial(text: str, rng: random.Random, intensity: float = 0.02) -> str:
    """SpaceInfi + sparse Unicode-Homoglyph (intensity = Anteil der lateinischen Treffer)."""
    # SpaceInfi: zusaetzliches Leerzeichen vor Komma (5 % der Kommata)
    def add_space(m):
        if rng.random() < 0.05:
            return " " + m.group(0)
        return m.group(0)
    out = re.sub(r",", add_space, text)
    # Homoglyph-Tausch
    chars = list(out)
    for i, ch in enumerate(chars):
        if ch.lower() in HOMOGLYPHS and rng.random() < intensity:
            sub = HOMOGLYPHS[ch.lower()]
            chars[i] = sub.upper() if ch.isupper() else sub
    return "".join(chars)


# ── Pangram ────────────────────────────────────────────────────────────────
async def score(pc: PangramClient, item_id: str, text: str) -> dict | None:
    res = await pc.check_bulk([{"id": item_id, "text": text}])
    r = res.get(item_id)
    if not r or r.error or r.fraction_ai is None:
        return None
    return {
        "fraction_ai": r.fraction_ai,
        "fraction_ai_assisted": r.fraction_ai_assisted,
        "fraction_human": r.fraction_human,
        "prediction": r.prediction,
    }


async def process_article(art: dict, human_pool: list[str], pc: PangramClient,
                           idx: int) -> dict:
    rng = random.Random(SEED + idx)
    orig = art["volltext"]
    base_id = f"hybrid_{idx:02d}_{art['doc_id']}"
    history = [{
        "stage": "baseline",
        "fraction_ai": art["pai_pre"], "fraction_ai_assisted": art["paa_pre"],
        "fraction_human": art["phu_pre"], "prediction": art["pred_pre"],
        "text_chars": len(orig),
    }]

    # Stage 1: Hybrid-Edit (Sentence-Mix)
    hybrid = inject_human_sentences(orig, human_pool, rng)
    s1 = await score(pc, f"{base_id}_hybrid", hybrid)
    if s1:
        history.append({"stage": "hybrid_edit", **s1, "text_chars": len(hybrid)})

    # Stage 2: + Mikro-Adversarial (intensity 0.02)
    micro = micro_adversarial(hybrid, rng, intensity=0.02)
    s2 = await score(pc, f"{base_id}_micro_low", micro)
    if s2:
        history.append({"stage": "hybrid_edit+micro_0.02", **s2, "text_chars": len(micro)})

    # Stage 3: + staerkere Mikro-Adversarial (0.05)
    micro_strong = micro_adversarial(hybrid, rng, intensity=0.05)
    s3 = await score(pc, f"{base_id}_micro_high", micro_strong)
    if s3:
        history.append({"stage": "hybrid_edit+micro_0.05", **s3, "text_chars": len(micro_strong)})

    # Best ueber alle Stages
    scored = [h for h in history[1:] if "fraction_ai" in h]
    best = min(scored, key=lambda h: h["fraction_ai"]) if scored else None
    if best == history[1]: best_text = hybrid
    elif scored and best == scored[1]: best_text = micro
    elif scored and len(scored) >= 3 and best == scored[2]: best_text = micro_strong
    else: best_text = hybrid

    return {
        "doc_id": art["doc_id"],
        "titel": art["titel"],
        "datum": art["datum"],
        "autor": art["autor"],
        "woerter": art["woerter"],
        "vorspann": art["vorspann"],
        "original_text": orig,
        "fraction_ai_pre": art["pai_pre"],
        "fraction_ai_post": best["fraction_ai"] if best else art["pai_pre"],
        "best_stage": best["stage"] if best else "?",
        "success": (best["fraction_ai"] < THRESHOLD) if best else False,
        "final_text": best_text,
        "history": history,
        "human_sentences_injected": (len(split_sentences(orig)) // HUMAN_INJECT_EVERY),
    }


async def main():
    print(f"--- Hybrid-Edit (Strategie 05) + Mikro-Adversarial (06) ---", flush=True)
    pool = load_human_pool()
    print(f"    Casdorff-Human-Saetze-Pool: {len(pool)}", flush=True)
    if not pool:
        sys.exit("Pool leer — keine Casdorff-Human-Artikel in DB.")

    arts = load_ai_articles(LIMIT)
    print(f"    Test-Artikel (AI eingestuft): {len(arts)} (LIMIT={LIMIT})", flush=True)

    results: list[dict] = []
    n_success = 0
    t_start = time.time()

    async with PangramClient() as pc:
        for i, art in enumerate(arts):
            r = await process_article(art, pool, pc, i)
            results.append(r)
            if r["success"]:
                n_success += 1
            elapsed = time.time() - t_start
            tag = "OK" if r["success"] else "FAIL"
            print(f"  [{i+1}/{len(arts)}] {tag:>4} {art['datum']} "
                  f"P(AI) {r['fraction_ai_pre']:.2f} -> {r['fraction_ai_post']:.3f} "
                  f"(best: {r['best_stage']}, inj {r['human_sentences_injected']} satz) "
                  f"[{n_success}/{i+1} success, {elapsed:.0f}s]",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== FERTIG Hybrid-Edit + Mikro ===")
    print(f"  Erfolg: {n_success}/{len(results)} ({n_success/max(len(results),1):.0%})")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
