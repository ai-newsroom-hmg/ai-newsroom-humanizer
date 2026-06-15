"""Humanize-Strategie 03 — Recursive Multi-Model Paraphrasing.

Theorie (Sadasivan et al., ICML 2024): selber Text wandert durch mehrere
LLM-Familien. Jeder Pass bricht die Perplexity-Signatur des vorherigen Modells.
Bricht Watermark-Detektoren 99,3 % -> 9,7 % gegen Pangram noch nicht publiziert.

Modell-Kette pro Artikel:
  Sonnet -> GPT-4o -> Gemini -> Sonnet
  (Anthropic -> OpenAI -> Google -> Anthropic)

Pangram nach jedem Pass. Bestmoegliche Variante speichern. Stop wenn P(AI) < 0.2.
Im Gegensatz zum reinen Sonnet-Loop hier KEIN Best-of-N — wir wollen Modell-
Diversitaet, nicht Variation innerhalb eines Modells.

Input:  alle Casdorff-Artikel die im 01-Loop NICHT erfolgreich waren (24 Artikel).
Output: data/humanize/03_multimodel_results.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _openrouter import ORClient  # noqa: E402

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
sys.path.insert(0, str(Path.home() / "Projects" / "ki-check" / "src"))
from ki_check.pangram import PangramClient  # noqa: E402

ROOT = Path.home() / "Projects" / "ki-check"
LOOP_RESULTS = ROOT / "data" / "humanize" / "01_casdorff_loop_results.jsonl"
OUT = ROOT / "data" / "humanize" / "03_multimodel_results.jsonl"

# Modell-Kette: Anthropic -> OpenAI -> Google -> Anthropic
MODEL_CHAIN = [
    ("anthropic/claude-sonnet-4.5",  "Sonnet"),
    ("openai/gpt-4o-2024-11-20",     "GPT-4o"),
    ("google/gemini-2.0-flash-001",  "Gemini-2.0"),
    ("anthropic/claude-sonnet-4.5",  "Sonnet (final)"),
]

THRESHOLD = 0.20
PARALLEL = 2
# Pilot-Limit (None = alle). Bei Pangram-Budget-Vorsicht klein anfangen.
LIMIT = int(os.environ.get("HUMANIZE_LIMIT", "5"))

SYSTEM_PROMPT = """Du bist ein erfahrener deutscher Journalist. Schreibe den
folgenden politischen Kommentar im Stil eines Tagesspiegel-Mitherausgebers um —
authentisch menschlich, mit allen Tells eines echten Journalisten. Halte
Aussage, Standpunkt, Quellen und Argumentationskette des Originals exakt ein.
Kein Inhalt darf hinzugefuegt oder entfernt werden. Eigennamen, Zahlen, Daten,
Zitate bleiben WORTGLEICH.

Was menschliche Stimme ausmacht:
- variable Satzlaengen, scharfe Brueche, Halbsatz als Pointe
- konkrete Namen, Daten, Beispiele
- idiomatische Wendungen, regionale Nuancen, etwas Schiefes
- keine zaehe Aufzaehlung mit drei parallelen Adjektiven
- keine 'Erstens / Zweitens / Drittens'-Struktur
- keine Floskel-Anschluesse ('daher', 'in diesem Kontext', 'vor diesem Hintergrund')
- Rhetorische Figuren ueberzeichnen statt glaetten

Antwortformat: NUR der umgeschriebene Text. Keine Vorrede, keine Erklaerung,
kein Titel, keine Markdown-Header."""


USER_TMPL = """ARTIKEL:

{text}

Schreibe ihn radikal menschlich um. Halte alle Fakten, Namen, Zahlen, Zitate
exakt ein. Eigene rhetorische Gestaltung erlaubt — Aussagegehalt bleibt
identisch."""


def load_input() -> list[dict]:
    """Lade alle Artikel, die im 01-Loop NICHT erfolgreich waren (oder alle, fuer Vergleich)."""
    if not LOOP_RESULTS.exists():
        sys.exit(f"INP fehlt: {LOOP_RESULTS}")
    rows = [json.loads(l) for l in LOOP_RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    valid = []
    for r in rows:
        if "error" in r:
            continue
        if not r.get("original_text") or len(r["original_text"]) < 500:
            continue
        valid.append(r)
    return valid


async def pangram_score(pc: PangramClient, item_id: str, text: str) -> dict | None:
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


async def humanize_multimodel(article: dict, idx: int, pc: PangramClient) -> dict:
    orig = article["original_text"]
    base_id = f"mm_{idx:02d}_{article['doc_id']}"

    history = [{
        "iter": 0,
        "model": "baseline",
        "fraction_ai": article.get("fraction_ai_pre", 1.0),
        "fraction_ai_assisted": 0.0,
        "fraction_human": 0.0,
        "prediction": "AI",
        "text_chars": len(orig),
    }]

    best_text = None
    best_pai = 1.0
    current_text = orig
    t0 = time.time()
    total_cost = 0.0

    for it, (model_slug, model_label) in enumerate(MODEL_CHAIN, start=1):
        # Paraphrase
        client = ORClient(model=model_slug)
        try:
            out = await client.complete(
                SYSTEM_PROMPT,
                USER_TMPL.format(text=current_text),
                temperature=0.8, max_tokens=3000,
            )
        except Exception as e:
            history.append({"iter": it, "model": model_label, "error": str(e)[:200]})
            continue
        new_text = out["text"].strip()
        total_cost += out["cost_usd"]

        # Pangram
        score = await pangram_score(pc, f"{base_id}_it{it}", new_text)
        if not score:
            history.append({"iter": it, "model": model_label,
                            "error": "pangram-no-score", "text_chars": len(new_text)})
            current_text = new_text
            continue

        history.append({
            "iter": it,
            "model": model_label,
            "model_slug": model_slug,
            "fraction_ai": score["fraction_ai"],
            "fraction_ai_assisted": score["fraction_ai_assisted"],
            "fraction_human": score["fraction_human"],
            "prediction": score["prediction"],
            "text_chars": len(new_text),
            "cost_usd": round(out["cost_usd"], 6),
        })

        # Best halten
        if score["fraction_ai"] < best_pai:
            best_pai = score["fraction_ai"]
            best_text = new_text

        current_text = new_text  # naechster Pass arbeitet auf dieser Version
        if best_pai < THRESHOLD:
            break

    return {
        "doc_id": article["doc_id"],
        "titel": article.get("titel"),
        "datum": article.get("datum"),
        "autor": article.get("autor"),
        "woerter": article.get("woerter"),
        "vorspann": article.get("vorspann"),
        "original_text": orig,
        "fraction_ai_pre": article.get("fraction_ai_pre", 1.0),
        "fraction_ai_post": best_pai,
        "iterations_run": len(history) - 1,
        "success": best_pai < THRESHOLD,
        "best_text": best_text if best_text else current_text,
        "final_text": best_text if best_text else current_text,
        "history": history,
        "total_cost_usd": round(total_cost, 4),
        "duration_s": round(time.time() - t0, 1),
    }


async def main():
    articles = load_input()
    if LIMIT:
        articles = articles[:LIMIT]
    print(f"--- Multi-Model Recursive Paraphrase: {len(articles)} Artikel (Pilot LIMIT={LIMIT}) ---", flush=True)
    print(f"    Modell-Kette: {[m[1] for m in MODEL_CHAIN]}", flush=True)
    print(f"    Threshold P(AI) < {THRESHOLD}, parallel {PARALLEL}", flush=True)

    sem = asyncio.Semaphore(PARALLEL)
    results = [None] * len(articles)
    done = 0
    t_start = time.time()
    total_cost = 0.0
    n_success = 0

    async with PangramClient() as pc:
        async def worker(i: int, art: dict):
            nonlocal done, total_cost, n_success
            async with sem:
                r = await humanize_multimodel(art, i, pc)
                results[i] = r
                done += 1
                total_cost += r.get("total_cost_usd", 0)
                if r.get("success"):
                    n_success += 1
                elapsed = time.time() - t_start
                tag = "OK" if r.get("success") else "FAIL"
                print(f"  [{done}/{len(articles)}] {tag:>4} {art.get('datum','?')} "
                      f"P(AI) {r['fraction_ai_pre']:.2f} -> {r['fraction_ai_post']:.3f} "
                      f"({r['iterations_run']} iter) "
                      f"({n_success}/{done} success, {elapsed:.0f}s, {total_cost:.2f}USD)",
                      flush=True)

        await asyncio.gather(*(worker(i, a) for i, a in enumerate(articles)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== FERTIG Multi-Model ===")
    print(f"  Artikel:        {len(results)}")
    print(f"  Erfolg (<{THRESHOLD}): {n_success}/{len(results)} ({n_success/max(len(results),1):.0%})")
    print(f"  Kosten total:   {total_cost:.4f} USD")
    print(f"  Output:         {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
