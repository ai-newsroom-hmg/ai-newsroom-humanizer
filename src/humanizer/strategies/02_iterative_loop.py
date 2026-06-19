"""Humanize-Strategie 02 — Iterativer Sonnet+Pangram-Loop (AuthorMist-light).

Pro Artikel:
  Iteration 0:  Pangram-Baseline auf Original
  Iteration k:  Sonnet bekommt {Original, aktuelle Variante, P(AI)_aktuell} und
                schreibt N Varianten -> kleinster P(AI)-Score gewinnt
  Stop:         P(AI) < THRESHOLD oder max MAX_ITERS

Input:  alle als AI klassifizierten Casdorff-Artikel aus ki-check.db
Output: data/humanize/01_casdorff_loop_results.jsonl
        ~/Downloads/ki-check-humanize-test/01-loop/ (Word-Files pre/post)
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "Projects" / "ki-check" / "src"))
from _openrouter import MODEL_DEFAULT, ORClient  # noqa: E402

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
from ki_check.pangram import PangramClient  # noqa: E402

ROOT = Path.home() / "Projects" / "ki-check"
DB = ROOT / "data" / "ki-check.db"
OUT = ROOT / "data" / "humanize" / "01_casdorff_loop_results.jsonl"

# Parameter
AUTHOR_LIKE = "%Casdorff%"
THRESHOLD = 0.20          # ziel: fraction_ai < 0.2
MAX_ITERS = 5
VARIANTS_PER_ITER = 3     # Best-of-N pro Runde
PARALLEL_ARTICLES = 2     # Sonnet rate-limit-freundlich
MODEL = MODEL_DEFAULT     # anthropic/claude-sonnet-4.5

SYSTEM_PROMPT = """Du bist ein erfahrener deutscher Journalist und Redakteur.
Deine Aufgabe: einen Text in der Stimme des Originalautors so umschreiben, dass
er authentisch menschlich klingt, mit allen Tells eines echten Journalisten —
nicht maschinenglatt. Halte Aussage, Standpunkt, Quellen und Argumentationskette
des Originals exakt ein. Kein Inhalt darf hinzugefuegt oder entfernt werden.

Was menschliche Stimme ausmacht (lt. Forschungsliteratur zu AI-Detektion):
- variable Satzlaengen, Mischung aus langen und sehr kurzen Saetzen
- gelegentliche Einschuebe, Gedankenspruenge, halbgar abgeschlossene Gedanken
- konkrete Namen, Daten, Beispiele statt Abstraktion
- idiomatische Wendungen, regionale Nuancen, etwas Schiefes
- keine zaehe Aufzaehlung mit drei parallelen Adjektiven
- keine "Erstens / Zweitens / Drittens"-Struktur
- keine Floskel-Anschluesse ("daher", "in diesem Kontext", "vor diesem Hintergrund")
- Brueche im Rhythmus erlaubt — Punkt statt Komma, Halbsatz als Pointe
- Originale rhetorische Figuren ueberzeichnen oder zerlegen statt glaetten

Antwortformat: NUR der umgeschriebene Text. Keine Vorrede, keine Erklaerung,
keine Markdown-Header, kein Titel."""


def user_prompt(original: str, current: str | None, pai_current: float | None,
                 iteration: int) -> str:
    if iteration == 0 or current is None:
        return (
            "ORIGINAL-ARTIKEL:\n\n"
            f"{original}\n\n"
            "Schreibe ihn so um, dass er menschlich klingt — Stil eines erfahrenen "
            "deutschen Politikkolumnisten. Halte alle Fakten, Namen, Aussagen exakt. "
            "Brich KI-typische Muster (Parallel-Listen, Floskel-Anschluesse, glatte "
            "Aufzaehlungen)."
        )
    return (
        f"ORIGINAL-ARTIKEL:\n\n{original}\n\n"
        f"AKTUELLE UMSCHRIFT (P(AI) = {pai_current:.3f}, Iteration {iteration}):\n\n"
        f"{current}\n\n"
        "Diese Umschrift wird vom KI-Detektor noch immer als KI-Text erkannt. "
        "Schreibe sie noch radikaler menschlich um: mehr Satz-Asymmetrie, mehr "
        "konkrete Details (Namen, Orte, Zeitstempel), mehr gewollte Brueche im "
        "Fluss, eigene Wendungen statt glatter Standardformeln. Inhalt bleibt "
        "exakt gleich; Stil aendert sich substantiell."
    )


def load_articles(limit: int | None = None) -> list[dict]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    q = """
        SELECT a.doc_id, a.datum, a.titel, a.autor, a.woerter, a.vorspann, a.volltext,
               c.fraction_ai AS pai_pre, c.fraction_ai_assisted AS paa_pre,
               c.fraction_human AS phu_pre, c.prediction AS pred_pre
        FROM article a
        JOIN check_result c ON c.doc_id = a.doc_id
        WHERE a.autor LIKE ?
          AND c.prediction = 'AI'
          AND a.volltext IS NOT NULL
          AND LENGTH(a.volltext) > 800
        GROUP BY a.doc_id
        ORDER BY c.fraction_ai DESC, a.datum DESC
    """
    if limit:
        q += f" LIMIT {limit}"
    rows = [dict(r) for r in con.execute(q, (AUTHOR_LIKE,))]
    con.close()
    return rows


async def pangram_score(pc: PangramClient, item_id: str, text: str) -> dict:
    """Single Pangram-Call. Liefert fraction_ai etc oder error."""
    results = await pc.check_bulk([{"id": item_id, "text": text}])
    r = results.get(item_id)
    if not r or r.error:
        return {"error": r.error if r else "no result"}
    return {
        "fraction_ai": r.fraction_ai,
        "fraction_ai_assisted": r.fraction_ai_assisted,
        "fraction_human": r.fraction_human,
        "prediction": r.prediction,
    }


async def humanize_one(art: dict, or_client: ORClient, pg_client: PangramClient,
                        idx: int) -> dict:
    """Einer Artikel komplett durch den Loop."""
    orig = art["volltext"]
    base_id = f"casdorff_{idx:02d}_{art['doc_id']}"

    history = []
    current_text = None
    current_pai = float(art["pai_pre"] or 1.0)

    history.append({
        "iter": 0,
        "fraction_ai": current_pai,
        "fraction_ai_assisted": float(art["paa_pre"] or 0.0),
        "fraction_human": float(art["phu_pre"] or 0.0),
        "prediction": art["pred_pre"],
        "text_chars": len(orig),
        "source": "baseline_db",
    })

    t0 = time.time()
    for it in range(1, MAX_ITERS + 1):
        if current_pai < THRESHOLD:
            break

        # Best-of-N: VARIANTS_PER_ITER Varianten parallel, beste nach Pangram nehmen
        variant_tasks = []
        for v in range(VARIANTS_PER_ITER):
            uprompt = user_prompt(orig, current_text, current_pai, it - 1)
            variant_tasks.append(or_client.complete(
                SYSTEM_PROMPT, uprompt, temperature=0.85, max_tokens=3000,
            ))
        variants = []
        try:
            outs = await asyncio.gather(*variant_tasks)
            for v_idx, o in enumerate(outs):
                variants.append({"v_idx": v_idx, "text": o["text"], "cost": o["cost_usd"]})
        except Exception as e:
            return {**_meta(art), "error": f"sonnet variant gen: {e!r}", "history": history,
                    "duration_s": round(time.time() - t0, 1)}

        # Pangram fuer alle Varianten parallel
        check_items = [{"id": f"{base_id}_it{it}_v{v['v_idx']}", "text": v["text"]} for v in variants]
        pg_res = await pg_client.check_bulk(check_items)

        # Beste Variante (kleinstes fraction_ai)
        scored = []
        for v in variants:
            r = pg_res.get(f"{base_id}_it{it}_v{v['v_idx']}")
            if not r or r.error or r.fraction_ai is None:
                continue
            scored.append((r.fraction_ai, r, v))
        if not scored:
            return {**_meta(art), "error": "pangram lieferte keine score", "history": history,
                    "duration_s": round(time.time() - t0, 1)}
        scored.sort(key=lambda x: x[0])
        best_pai, best_r, best_v = scored[0]

        current_text = best_v["text"]
        current_pai = best_pai
        history.append({
            "iter": it,
            "fraction_ai": best_pai,
            "fraction_ai_assisted": best_r.fraction_ai_assisted,
            "fraction_human": best_r.fraction_human,
            "prediction": best_r.prediction,
            "text_chars": len(current_text),
            "variants_tested": len(scored),
            "variant_costs_usd": round(sum(v["cost"] for v in variants), 6),
            "source": "sonnet_loop",
        })

    return {
        **_meta(art),
        "doc_id": art["doc_id"],
        "iterations_run": len(history) - 1,
        "success": current_pai < THRESHOLD,
        "fraction_ai_pre": float(art["pai_pre"] or 1.0),
        "fraction_ai_post": current_pai,
        "final_text": current_text or orig,
        "history": history,
        "total_cost_usd": round(sum(h.get("variant_costs_usd", 0) for h in history), 4),
        "duration_s": round(time.time() - t0, 1),
    }


def _meta(art: dict) -> dict:
    return {
        "datum": art["datum"],
        "titel": art["titel"],
        "autor": art["autor"],
        "woerter": art["woerter"],
        "vorspann": art["vorspann"],
        "original_text": art["volltext"],
    }


async def main():
    rows = load_articles()
    print(f"--- Humanize-Loop (Strategie 02): {len(rows)} Casdorff-AI-Artikel ---", flush=True)
    print(f"    Threshold P(AI) < {THRESHOLD}, max {MAX_ITERS} Iter, "
          f"{VARIANTS_PER_ITER} Varianten/Iter, parallel {PARALLEL_ARTICLES} Artikel", flush=True)

    or_client = ORClient(model=MODEL)
    sem = asyncio.Semaphore(PARALLEL_ARTICLES)
    results: list[dict] = [None] * len(rows)
    done = 0
    t_start = time.time()
    total_cost = 0.0
    n_success = 0

    async with PangramClient() as pg_client:
        async def worker(i: int, art: dict):
            nonlocal done, total_cost, n_success
            async with sem:
                r = await humanize_one(art, or_client, pg_client, i)
                results[i] = r
                done += 1
                if "error" not in r:
                    total_cost += r.get("total_cost_usd", 0)
                    if r.get("success"):
                        n_success += 1
                elapsed = time.time() - t_start
                pai_post = r.get("fraction_ai_post")
                iters = r.get("iterations_run", 0)
                tag = "OK" if r.get("success") else ("ERR" if "error" in r else "FAIL")
                print(f"  [{done}/{len(rows)}] {tag:>4} {art['datum']} "
                      f"P(AI) 1.000 -> {pai_post:.3f} after {iters} iter "
                      f"(success {n_success}/{done}, elapsed {elapsed:.0f}s, "
                      f"cost {total_cost:.2f} USD)", flush=True)

        await asyncio.gather(*(worker(i, r) for i, r in enumerate(rows)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in results if r and "error" not in r)
    n_err = len(results) - n_ok
    print("\n=== FERTIG ===")
    print(f"  Artikel:        {len(results)}")
    print(f"  Erfolg (<{THRESHOLD}): {n_success}/{n_ok}")
    print(f"  Fehler:         {n_err}")
    print(f"  Kosten total:   {total_cost:.4f} USD")
    print(f"  Output:         {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
