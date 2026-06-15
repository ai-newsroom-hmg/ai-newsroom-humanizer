"""Phase A.2 — Stilometric Loop mit Casdorff-Stil-Profil im Prompt.

Pfad A aus der State-of-Art-Recherche: Sonnet bekommt das aus den 28 Casdorff-
Human-Artikeln extrahierte Stil-Profil als System-Prompt-Stuetze. Wir glauben,
dass authentische Stil-Signale (Satzlaengen-Verteilung, Vokabular, Anti-Pattern)
Pangram-Tells brechen.

Iterativ wie Strategie 02, aber Anweisungen werden konkret: nicht 'menschlich
schreiben', sondern 'P95 Satzlaenge 31 Tokens, 12% Saetze unter 5 Tokens,
diese Vokabel-Marker, kein "daher/in diesem Kontext"'.

Pilot: 5 Casdorff-AI-Artikel (gleiche wie humanize_01 Top-5).
Output: data/humanize/07_stilometric_results.jsonl
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
from _openrouter import ORClient, MODEL_DEFAULT  # noqa: E402

if not os.environ.get("PANGRAM_API_KEY"):
    f = Path.home() / ".config" / "pangram" / "key"
    if f.exists():
        os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
sys.path.insert(0, str(Path.home() / "Projects" / "ki-check" / "src"))
from ki_check.pangram import PangramClient  # noqa: E402

ROOT = Path.home() / "Projects" / "ki-check"
DB = ROOT / "data" / "ki-check.db"
PROFILE = ROOT / "data" / "humanize" / "casdorff_profile.json"
OUT = ROOT / "data" / "humanize" / "07_stilometric_results.jsonl"

THRESHOLD = 0.20
MAX_ITERS = 5
LIMIT = int(os.environ.get("HUMANIZE_LIMIT", "5"))
PARALLEL = 2
MODEL = MODEL_DEFAULT


def build_system_prompt(profile: dict) -> str:
    sl = profile["satzlaengen"]
    dist = profile["satz_distribution_pct"]
    rh = profile["rhetorik"]
    starters = profile["haeufige_satzanfaenge"][:15]
    vocab = profile["casdorff_vokabular_top30"][:20]
    phrases = profile.get("wiederkehrende_phrasen_3g", [])[:10]
    examples = profile["beispiel_eroeffnungen_15"][:5]
    antis = profile["anti_pattern"]

    return f"""Du bist Stephan-Andreas Casdorff, Mitherausgeber des Tagesspiegel. Du
schreibst politische Kommentare und Glossen. Dein Stil ist aus deinem realen
Korpus von 28 Artikeln (1117 Saetzen) gemessen worden.

DEINE EMPIRISCHE STIL-SIGNATUR (Pflicht zu treffen):

Satzlaengen (Tokens pro Satz):
- Mean {sl['mean']}, Median {sl['median']}, P95 {sl['p95']}, Max {sl['max']}
- {dist.get('1-5', 0)} % der Saetze sind 1-5 Tokens kurz (Pointen, Schlagworte)
- {dist.get('6-10', 0)} % 6-10 Tokens (Hauptmasse)
- {dist.get('11-15', 0)} % 11-15 Tokens
- {dist.get('21-25', 0) + dist.get('26-30', 0) + dist.get('>30', 0):.1f} % > 20 Tokens

Rhetorisches Profil:
- {rh['fragen_quote_pct']} % der Saetze sind Fragen (rhetorisch, treibend)
- {rh['ausruf_quote_pct']} % Ausrufe (selten, gezielt eingesetzt)
- Kurzsatz-Anteil {rh['kurzsatz_anteil_pct']} % — viele knappe Pointen
- Langsatz-Anteil {rh['langsatz_anteil_pct']} % — wenige Satz-Architekturen

Haeufige Satzanfaenge (verwende mehrfach):
{', '.join(starters)}

Casdorff-Vokabular (typisch in deinen Texten):
{', '.join(vocab)}

Wiederkehrende Phrasen (3-Wort-Cluster):
{chr(10).join('- ' + p for p in phrases)}

Beispiel-Eroeffnungen aus deinem Korpus:
{chr(10).join('- "' + e + '"' for e in examples)}

ANTI-PATTERN (NIEMALS schreiben — KI-Tells):
{chr(10).join('- ' + a for a in antis)}

Was du UNBEDINGT tust:
- variable Satzlaengen mit Pointen
- Halbsatz als Pointe, Punkt statt Komma fuer Wirkung
- Idiomatische Wendungen, sprachlich konkrete Details
- Rhetorische Frage einbauen wenn passend
- Eigennamen, Daten, Zahlen exakt
- Kein Markdown, kein Header, kein Titel

Was du NIEMALS tust:
- Floskel-Anschluesse 'daher', 'in diesem Kontext', 'vor diesem Hintergrund'
- Erstens / Zweitens / Drittens
- Drei parallele Adjektive ('klar, transparent, nachvollziehbar')
- Glatte Aufzaehlung ohne Bruch
- Markdown / Listen / Headings
- AI-Vorrede wie 'Es ist wichtig zu erwaehnen'"""


USER_TMPL = """ORIGINAL-KOMMENTAR (vom KI-Detektor als KI klassifiziert):

{text}

Schreibe ihn neu — in DEINEM Stil (Casdorff). Alle Fakten, Namen, Zahlen,
Zitate exakt erhalten. Inhalt nicht aendern. Nur Stil dehnen, Saetze brechen,
Pointen setzen.

Antwortformat: NUR den umgeschriebenen Text. Keine Vorrede."""


USER_ITER_TMPL = """ORIGINAL:

{orig}

LETZTE FASSUNG (P(AI) = {pai:.3f} — noch immer KI-klassifiziert):

{current}

Schreibe noch radikaler in deinem Stil. Mehr Bruch, mehr Pointe, mehr Asymmetrie.
Inhalt absolut treu zum Original. NUR Stil aendern."""


def load_articles(limit: int) -> list[dict]:
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


async def humanize_one(art: dict, system: str, client: ORClient, pc: PangramClient,
                        idx: int) -> dict:
    orig = art["volltext"]
    base_id = f"stil_{idx:02d}_{art['doc_id']}"
    history = [{
        "iter": 0,
        "fraction_ai": art["pai_pre"], "fraction_ai_assisted": art["paa_pre"],
        "fraction_human": art["phu_pre"], "prediction": art["pred_pre"],
        "text_chars": len(orig), "source": "baseline_db",
    }]

    current_text = None
    current_pai = float(art["pai_pre"])
    t0 = time.time()
    total_cost = 0.0

    for it in range(1, MAX_ITERS + 1):
        if current_pai < THRESHOLD:
            break
        if it == 1:
            user = USER_TMPL.format(text=orig)
        else:
            user = USER_ITER_TMPL.format(orig=orig, pai=current_pai, current=current_text)

        try:
            out = await client.complete(system, user, temperature=0.85, max_tokens=3000)
        except Exception as e:
            history.append({"iter": it, "error": str(e)[:200]})
            break
        new_text = out["text"].strip()
        total_cost += out["cost_usd"]

        res = await pc.check_bulk([{"id": f"{base_id}_it{it}", "text": new_text}])
        r = res.get(f"{base_id}_it{it}")
        if not r or r.error or r.fraction_ai is None:
            history.append({"iter": it, "error": "pangram-no-score", "text_chars": len(new_text)})
            current_text = new_text
            continue

        history.append({
            "iter": it,
            "fraction_ai": r.fraction_ai,
            "fraction_ai_assisted": r.fraction_ai_assisted,
            "fraction_human": r.fraction_human,
            "prediction": r.prediction,
            "text_chars": len(new_text),
            "cost_usd": round(out["cost_usd"], 6),
            "source": "stilometric_loop",
        })
        current_text = new_text
        current_pai = r.fraction_ai

    return {
        "doc_id": art["doc_id"],
        "titel": art["titel"],
        "datum": art["datum"],
        "autor": art["autor"],
        "woerter": art["woerter"],
        "vorspann": art["vorspann"],
        "original_text": orig,
        "fraction_ai_pre": float(art["pai_pre"]),
        "fraction_ai_post": current_pai,
        "iterations_run": len(history) - 1,
        "success": current_pai < THRESHOLD,
        "final_text": current_text or orig,
        "history": history,
        "total_cost_usd": round(total_cost, 4),
        "duration_s": round(time.time() - t0, 1),
    }


async def main():
    if not PROFILE.exists():
        sys.exit(f"Profile fehlt: {PROFILE} — laeuft humanize_06 zuerst")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    system = build_system_prompt(profile)
    arts = load_articles(LIMIT)
    print(f"--- Stilometric Loop (Pfad A) ---", flush=True)
    print(f"    Profil: {profile['korpus_n']} Human-Artikel, {profile['saetze_total']} Saetze", flush=True)
    print(f"    Test: {len(arts)} Casdorff-AI-Artikel, max {MAX_ITERS} Iter, T={THRESHOLD}", flush=True)

    client = ORClient(model=MODEL)
    sem = asyncio.Semaphore(PARALLEL)
    results = [None] * len(arts)
    done = 0
    n_success = 0
    t_start = time.time()
    total_cost = 0.0

    async with PangramClient() as pc:
        async def worker(i: int, art: dict):
            nonlocal done, total_cost, n_success
            async with sem:
                r = await humanize_one(art, system, client, pc, i)
                results[i] = r
                done += 1
                total_cost += r.get("total_cost_usd", 0)
                if r.get("success"):
                    n_success += 1
                elapsed = time.time() - t_start
                tag = "OK" if r.get("success") else "FAIL"
                print(f"  [{done}/{len(arts)}] {tag:>4} {art['datum']} "
                      f"P(AI) {r['fraction_ai_pre']:.2f} -> {r['fraction_ai_post']:.3f} "
                      f"({r['iterations_run']} iter, {r['duration_s']:.0f}s, {r['total_cost_usd']:.3f}USD) "
                      f"[{n_success}/{done} success, total {total_cost:.2f}USD]",
                      flush=True)

        await asyncio.gather(*(worker(i, a) for i, a in enumerate(arts)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== FERTIG Pfad A: Stilometric Prompt ===")
    print(f"  Erfolg: {n_success}/{len(results)} ({n_success/max(len(results),1):.0%})")
    print(f"  Kosten: {total_cost:.4f} USD")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
