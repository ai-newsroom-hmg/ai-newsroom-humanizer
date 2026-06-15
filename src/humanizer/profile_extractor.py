"""Phase A.1 — Casdorff-Stil-Profil aus 28 Human-Artikeln extrahieren.

Out: data/humanize/casdorff_profile.json

Felder:
  satzlaengen           — Mean, Median, P25, P75, P95 (Token-Zaehlung)
  satz_distribution     — Histogramm 1-5/6-10/11-15/16-20/21-25/26-30/>30
  haeufige_konnektoren  — Top-30 Anschluss-Adverbien/Konjunktionen am Satzanfang
  haeufige_oeffnungen   — Top-20 Erst-Saetze (3 Tokens)
  pangramatische_floskeln — Phrasen die er WIEDERHOLT nutzt (>=3 Vorkommen)
  rhetorische_figuren    — Frage-Satz-Quote, Ellipse, Apokoinu-Indikatoren
  vokabular_unique       — Casdorff-typische Woerter (TF-IDF top vs Stoppliste)
  anti_pattern           — Was Casdorff NIE schreibt (zaehe Parallel-Listen,
                            'erstens/zweitens/drittens', Floskel-Anschluesse)
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "Projects" / "ki-check"
DB = ROOT / "data" / "ki-check.db"
OUT = ROOT / "data" / "humanize" / "casdorff_profile.json"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\wäöüßÄÖÜ]+\b", text.lower(), flags=re.UNICODE)


def load_human_texts() -> list[str]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT a.volltext FROM article a
        JOIN check_result c ON c.doc_id=a.doc_id
        WHERE a.autor LIKE '%Casdorff%' AND c.prediction='Human'
          AND a.volltext IS NOT NULL
    """).fetchall()
    con.close()
    return [r["volltext"] for r in rows]


def main():
    texts = load_human_texts()
    print(f"--- Casdorff-Stil-Profil aus {len(texts)} Human-Artikeln ---")

    all_sentences = []
    for t in texts:
        all_sentences.extend(split_sentences(t))
    print(f"    Saetze gesamt: {len(all_sentences)}")

    # ─── Satzlaengen (in Tokens) ────────────────────────────────────────────
    lens = [len(tokenize(s)) for s in all_sentences if len(s) > 10]
    bins = Counter()
    for l in lens:
        if l <= 5: bins["1-5"] += 1
        elif l <= 10: bins["6-10"] += 1
        elif l <= 15: bins["11-15"] += 1
        elif l <= 20: bins["16-20"] += 1
        elif l <= 25: bins["21-25"] += 1
        elif l <= 30: bins["26-30"] += 1
        else: bins[">30"] += 1

    satz_distribution = {k: round(v / len(lens) * 100, 1) for k, v in bins.most_common()}

    satzlaengen = {
        "mean":   round(statistics.mean(lens), 1),
        "median": statistics.median(lens),
        "p25":    statistics.quantiles(lens, n=4)[0],
        "p75":    statistics.quantiles(lens, n=4)[2],
        "p95":    statistics.quantiles(lens, n=20)[18],
        "min":    min(lens),
        "max":    max(lens),
    }

    # ─── Anschluss-Konnektoren (erstes Wort der Saetze ab Satz 2) ──────────
    starters = Counter()
    starter_phrases = Counter()
    for s in all_sentences:
        toks = tokenize(s)
        if not toks: continue
        starters[toks[0]] += 1
        if len(toks) >= 3:
            starter_phrases[" ".join(toks[:3])] += 1

    haeufige_konnektoren = [w for w, c in starters.most_common(40) if c >= 5]

    # ─── Repetitive Phrasen (Bigramme + Trigramme >= 3 Vorkommen) ──────────
    bigrams = Counter()
    trigrams = Counter()
    for s in all_sentences:
        toks = tokenize(s)
        for i in range(len(toks) - 1):
            bigrams[(toks[i], toks[i+1])] += 1
        for i in range(len(toks) - 2):
            trigrams[(toks[i], toks[i+1], toks[i+2])] += 1

    # Filter: typische deutsche Floskeln raus
    STOPGRAMS = {("und", "der"), ("ist", "die"), ("auf", "die"), ("für", "die"),
                 ("in", "der"), ("an", "der"), ("zu", "den"), ("mit", "der")}
    casdorff_phrases = [
        " ".join(t) for t, c in trigrams.most_common(150)
        if c >= 3 and not any(t[i] in {"der", "die", "das", "ein", "und", "in", "auf"} for i in range(2))
    ][:20]

    # ─── Vokabular (Top-Woerter relativ zur deutschen Grundsprache) ───────
    word_freq = Counter()
    for s in all_sentences:
        for w in tokenize(s):
            if len(w) > 4:
                word_freq[w] += 1

    # Erst Filter: Stoppliste
    DE_STOP = {"deutschland", "menschen", "viele", "ihre", "ihrer", "deren", "sind", "haben",
                "werden", "geben", "kommen", "gehen", "wissen", "diese", "diesen", "diesem",
                "jedoch", "schon", "noch", "wieder", "immer", "mehr", "weniger", "ihren",
                "ihrem", "denen", "vielleicht"}

    vokabular_unique = [
        w for w, c in word_freq.most_common(80)
        if c >= 4 and w not in DE_STOP
    ][:30]

    # ─── Fragen-Quote ──────────────────────────────────────────────────────
    n_frage = sum(1 for s in all_sentences if s.endswith("?"))
    n_ausruf = sum(1 for s in all_sentences if s.endswith("!"))
    rhetorische = {
        "fragen_quote_pct":   round(n_frage / len(all_sentences) * 100, 1),
        "ausruf_quote_pct":   round(n_ausruf / len(all_sentences) * 100, 1),
        "kurzsatz_anteil_pct": round(bins.get("1-5", 0) / len(lens) * 100, 1),
        "langsatz_anteil_pct": round((bins.get("26-30", 0) + bins.get(">30", 0)) / len(lens) * 100, 1),
    }

    # ─── Erst-Saetze (Eroeffnungen) ────────────────────────────────────────
    eroeffnungen = []
    for t in texts:
        ss = split_sentences(t)
        if ss:
            eroeffnungen.append(ss[0][:150])
    eroeffnungen_sample = eroeffnungen[:15]

    profile = {
        "autor": "Stephan-Andreas Casdorff",
        "outlet": "Tagesspiegel",
        "korpus_n": len(texts),
        "saetze_total": len(all_sentences),

        "satzlaengen": satzlaengen,
        "satz_distribution_pct": satz_distribution,

        "rhetorik": rhetorische,

        "haeufige_satzanfaenge": haeufige_konnektoren,
        "haeufige_starter_phrases": [
            " ".join(p) for p, c in starter_phrases.most_common(20) if c >= 3
        ][:15],

        "wiederkehrende_phrasen_3g": casdorff_phrases,
        "casdorff_vokabular_top30": vokabular_unique,

        "beispiel_eroeffnungen_15": eroeffnungen_sample,

        "anti_pattern": [
            "Erstens / Zweitens / Drittens-Struktur",
            "Drei parallele Adjektive (z.B. 'klar, transparent, nachvollziehbar')",
            "Floskel-Anschluesse: 'daher', 'in diesem Kontext', 'vor diesem Hintergrund', 'zusammenfassend'",
            "Glatte Aufzaehlungen ohne Bruch",
            "Lange Listen mit gleicher Satzstruktur",
            "Markdown-Header / Aufzaehlungspunkte",
            "Korrekte Anfuehrungszeichen-Konsistenz (echte Texte mischen oft ,,'' und „``)",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"--- Profil geschrieben: {OUT}")
    print()
    print(f"Satzlaengen: Mean {satzlaengen['mean']}, Median {satzlaengen['median']},"
          f" P95 {satzlaengen['p95']}, max {satzlaengen['max']}")
    print(f"Verteilung: {satz_distribution}")
    print(f"Fragen-Quote: {rhetorische['fragen_quote_pct']} %")
    print(f"Kurzsaetze (1-5 Tokens): {rhetorische['kurzsatz_anteil_pct']} %")
    print(f"Langsaetze (>25 Tokens): {rhetorische['langsatz_anteil_pct']} %")
    print(f"Top-Vokabular (10): {vokabular_unique[:10]}")
    print(f"Top-Satzanfaenge (10): {haeufige_konnektoren[:10]}")


if __name__ == "__main__":
    main()
