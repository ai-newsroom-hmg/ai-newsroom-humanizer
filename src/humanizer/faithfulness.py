"""Humanize-Phase 02 — Inhalts-Treue-Check.

Pro humanisiertem Artikel: Sonnet vergleicht ORIGINAL gegen FINAL und gibt
strukturiertes Urteil zurueck — welche Aussagen / Namen / Zahlen sind erhalten,
welche fehlen, welche wurden hinzugefuegt.

Plus deterministische Strukturchecks:
- Eigennamen (Personen, Orte, Organisationen) Erhalt
- Zahlen / Daten Erhalt
- 4-gram-Overlap mit Original (zu hoch = nicht humanisiert; zu niedrig = Inhalt veraendert)

Input:  data/humanize/01_casdorff_loop_results.jsonl
Output: data/humanize/02_faithfulness.jsonl
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _openrouter import ORClient, MODEL_DEFAULT  # noqa: E402

ROOT = Path.home() / "Projects" / "ki-check"
INP = ROOT / "data" / "humanize" / "01_casdorff_loop_results.jsonl"
OUT = ROOT / "data" / "humanize" / "02_faithfulness.jsonl"

PARALLEL = 4

JUDGE_SYSTEM = """Du bist Fact-Checker eines Qualitaetsmagazins. Du bekommst zwei
Versionen desselben Artikels und pruefst, ob die zweite Version den INHALT der
ersten Version exakt bewahrt — nur Stil und Formulierung duerfen sich aendern.

Antworte als reines JSON mit diesem Schema:

{
  "faithfulness_score": <float 0..1>,    // 1.0 = inhaltlich identisch
  "preserved": ["Aussage 1", "Aussage 2", ...],  // erhaltene Kernaussagen
  "missing": ["Aussage X fehlt", ...],            // fehlende Aussagen
  "added": ["Aussage Y hinzugefuegt (nicht im Original)", ...],
  "name_changes": ["Original: 'Merz' -> Final: 'Friedrich Merz' (OK / NICHT OK)"],
  "number_changes": ["Original: '500 Mrd' -> Final: '500 Milliarden' (OK / NICHT OK)"],
  "verdict": "FAITHFUL | MINOR_DRIFT | SIGNIFICANT_DRIFT | CONTENT_CHANGED",
  "comment": "kurze Begruendung in 1-2 Saetzen"
}

Kriterien:
- FAITHFUL (>= 0.95): jede inhaltliche Aussage des Originals findet sich in der Final-Version, nichts hinzugefuegt
- MINOR_DRIFT (0.85-0.95): leichte Auslassungen oder Praezisierungen, aber kein Sinn-Verlust
- SIGNIFICANT_DRIFT (0.7-0.85): substanzielle Auslassungen ODER hinzugefuegter Inhalt
- CONTENT_CHANGED (<0.7): Sinn-Veraenderung, Falsche Zuschreibung, oder ganzer Strang fehlt

Sei streng. Stil-Aenderungen sind erlaubt — Inhalts-Aenderungen nicht."""


JUDGE_USER_TMPL = """ORIGINAL:

{orig}

FINAL (humanisierte Version):

{final}

Pruefe nach dem oben definierten Schema."""


# ── Deterministische Strukturchecks ────────────────────────────────────────
def extract_capitals(text: str) -> set[str]:
    """Mehrteilige Eigennamen + Einzelnamen via Heuristik."""
    # Multi-token Capitalized phrases (Namen / Organisationen)
    multi = set(re.findall(r"\b(?:[A-ZÄÖÜ][\wäöüß]+(?:[-\s][A-ZÄÖÜ][\wäöüß]+){1,4})\b", text))
    # Plus Einzel-Capitals (haeufig Akteure)
    single = set(re.findall(r"\b[A-ZÄÖÜ][\wäöüß]{3,}\b", text))
    return multi | single


def extract_numbers(text: str) -> set[str]:
    """Zahlen + Daten + Prozent (normiert)."""
    nums = set(re.findall(r"\b\d{1,4}(?:[.,]\d+)?\s*(?:%|Mrd|Mio|Prozent|Euro|Dollar|€|\$)?", text))
    dates = set(re.findall(r"\b\d{1,2}\.\s*(?:Januar|Februar|Maerz|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*\d{2,4}\b", text))
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    return nums | dates | years


def four_gram_overlap(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^\w]+", " ", s.lower(), flags=re.UNICODE).split()
    ta, tb = norm(a), norm(b)
    if len(ta) < 4 or len(tb) < 4:
        return 0.0
    ga = {tuple(ta[i:i + 4]) for i in range(len(ta) - 3)}
    gb = {tuple(tb[i:i + 4]) for i in range(len(tb) - 3)}
    return len(ga & gb) / max(len(ga), 1)


def structural_check(orig: str, final: str) -> dict:
    caps_o = extract_capitals(orig)
    caps_f = extract_capitals(final)
    nums_o = extract_numbers(orig)
    nums_f = extract_numbers(final)
    return {
        "names_lost":           sorted(caps_o - caps_f)[:30],
        "names_added":          sorted(caps_f - caps_o)[:30],
        "names_preserved_ratio": round(len(caps_o & caps_f) / max(len(caps_o), 1), 3),
        "numbers_lost":         sorted(nums_o - nums_f)[:30],
        "numbers_added":        sorted(nums_f - nums_o)[:30],
        "numbers_preserved_ratio": round(len(nums_o & nums_f) / max(len(nums_o), 1), 3),
        "four_gram_overlap":    round(four_gram_overlap(orig, final), 3),
    }


async def llm_judge(client: ORClient, orig: str, final: str) -> dict:
    user = JUDGE_USER_TMPL.format(orig=orig[:8000], final=final[:8000])
    out = await client.complete(JUDGE_SYSTEM, user, temperature=0.1, max_tokens=2000)
    text = out["text"].strip()
    # Try direct JSON
    try:
        return {"_llm_cost_usd": out["cost_usd"], **json.loads(text)}
    except Exception:
        pass
    # Try JSON innerhalb von Code-Block extrahieren
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return {"_llm_cost_usd": out["cost_usd"], **json.loads(m.group(0))}
        except Exception as e:
            return {"_llm_cost_usd": out["cost_usd"], "_parse_error": str(e)[:200], "_raw": text[:500]}
    return {"_llm_cost_usd": out["cost_usd"], "_parse_error": "no json found", "_raw": text[:500]}


async def main():
    if not INP.exists():
        sys.exit(f"INP fehlt: {INP}")
    rows = [json.loads(l) for l in INP.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [r for r in rows if "error" not in r and r.get("final_text")]
    print(f"--- Faithfulness-Check: {len(todo)} humanisierte Artikel ---", flush=True)

    client = ORClient(model=MODEL_DEFAULT)
    sem = asyncio.Semaphore(PARALLEL)
    results: list[dict] = [None] * len(todo)
    done = 0
    t_start = time.time()
    total_cost = 0.0

    async def worker(i: int, r: dict):
        nonlocal done, total_cost
        orig = r.get("original_text") or ""
        final = r.get("final_text") or ""
        struct = structural_check(orig, final)
        async with sem:
            try:
                judge = await llm_judge(client, orig, final)
            except Exception as e:
                judge = {"_error": str(e)[:200]}
            done += 1
            total_cost += judge.get("_llm_cost_usd", 0) or 0
            elapsed = time.time() - t_start
            print(f"  [{done}/{len(todo)}] {r.get('titel','')[:50]} "
                  f"| 4gram={struct['four_gram_overlap']:.3f} "
                  f"| names={struct['names_preserved_ratio']:.2f} "
                  f"| nums={struct['numbers_preserved_ratio']:.2f} "
                  f"| verdict={judge.get('verdict','?')} "
                  f"| {elapsed:.0f}s {total_cost:.3f}USD", flush=True)
            results[i] = {
                "doc_id": r["doc_id"],
                "titel": r.get("titel"),
                "datum": r.get("datum"),
                "fraction_ai_pre": r.get("fraction_ai_pre"),
                "fraction_ai_post": r.get("fraction_ai_post"),
                "iterations_run": r.get("iterations_run"),
                "humanize_success": r.get("success"),
                "structural": struct,
                "llm_judge": judge,
            }

    await asyncio.gather(*(worker(i, r) for i, r in enumerate(todo)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    verdicts = Counter((r["llm_judge"].get("verdict") or "?") for r in results if r)
    print(f"\n=== Faithfulness-Summary ===")
    for v, c in verdicts.most_common():
        print(f"  {v}: {c}")
    print(f"  Total LLM cost: {total_cost:.4f} USD")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
