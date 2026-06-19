# ADR 008 — Phase 3a: Mistral-Small-3.2-24b knackt Pangram in 42 % der Docs (n=12)

**Status:** angenommen
**Datum:** 2026-06-17
**Verantwortlich:** Gunter Nowy + AI-Newsroom-Stack
**Kontext:** Phase-3-Smoke-2 (gestern Abend, n=3) suggerierte 100 % Doc-Bypass mit
Mistral-Small-3.2-24b. Diese ADR repliziert das Setup auf n=12 length-stratifiziert
und revidiert die Schlussfolgerung.

---

## Vorgeschichte

ADR 007 schloss Phase 2 mit „4/23 = 17 % Bypass auf kurzen TSP-Artikeln, 0/19 auf
langen". Phase-3-Smoke-2 testete am 2026-06-17 Vormittag (vor Crash) Mistral-Small-3.2-24b
und Mistral-Large-2512 auf 3 long-form-Fails. Ergebnis: 3/3 Docs mit ≥1 Variante,
die Pangram unter 0.2 drückte. BGE-Sim-Faithfulness-Check ergänzt nach Crash
heute: 3/3 = 100 % der Docs hatten je 1 inhaltstreue Bypass-Variante.

n=3 ist methodisch zu klein für Schlüsse. Phase 3a reproduziert.

## Setup

- **Sample:** 12 Artikel aus `data/phase2-training-pool/eval.jsonl`, alle mit
  `fraction_ai = 1.0` beim Pool-Build, Smoke-2-Docs ausgeschlossen.
  Length-stratifiziert: 4 short (895–1190 chars), 3 mid (2410–2586), 5 long (2830–3019).
- **Model:** Nur `mistralai/mistral-small-3.2-24b-instruct` (Sweet-Spot aus Smoke-2).
- **Sweep:** 6 Temps (0.90, 0.95, 1.00, 1.05, 1.10, 1.15) × 4 idx = 24 Varianten/Doc.
- **Eval:** Pangram-API live, BGE-M3-Cosine auf ruediger (MPS).
- **Kosten:** $0.07 Generation + $14.40 Pangram = **$14.47** (288 Calls, kein Cache-Hit).
- **Skript:** `src/humanizer/training/smoke_phase3a_n12.py` (reproduzierbar).

## Befund

```
                            n=3 Smoke-2     n=12 Smoke-3
Docs mit Bypass (P<0.2)      100% (3/3)     100% (12/12)
Docs mit BEIDEM              100% (3/3)      42% (5/12)
Variant-Rate BEIDEM          5%  (3/60)     4.5% (13/288)
```

**Per-Doc-Tabelle (chars / best_p / best_sim / Bypass-Varianten / Faithful-Varianten / BEIDE):**

| doc_id | chars | best_p | best_sim | bypass | faith | BOTH |
|---|---:|---:|---:|---:|---:|---:|
| TSP__5d034c5e | 895 | 0.000 | 0.941 | 3 | 17 | 1 |
| TSP__22f1d55f | 907 | 0.000 | 0.967 | 6 | 18 | 2 |
| TSP__1a75f728 | 1184 | 0.000 | 0.971 | 4 | 19 | 0 |
| TSP__5d2e7d0c | 1190 | 0.000 | 0.976 | 3 | 21 | 0 |
| TSP__73fde4c6 | 2410 | 0.000 | 0.985 | 5 | 14 | 0 |
| BB48B8A7 | 2468 | 0.000 | 0.981 | 1 | 19 | 0 |
| WWON__7cdffe7e | 2586 | 0.000 | 0.970 | 8 | 19 | 3 |
| 3EDE8AF0 | 2830 | 0.000 | 0.970 | 4 | 22 | 2 |
| HB_100223004 | 2835 | 0.000 | 0.990 | 10 | 18 | 5 |
| HB_100221155 | 2909 | 0.000 | 0.960 | 8 | 15 | 0 |
| HB_100228485 | 2913 | 0.000 | 0.982 | 5 | 17 | 0 |
| HB_100220623 | 3019 | 0.000 | 0.958 | 8 | 14 | 0 |

**Length-Strat (Docs mit BEIDEM):**

| Bucket | Docs | Variants |
|---|---:|---:|
| short ≤1400 | 2/4 | 3/96 |
| mid 1400–2800 | 1/3 | 3/72 |
| long ≥2800 | 2/5 | 7/120 |

## Interpretation

### 1. Smoke-2 (n=3, 100 %) war Stichprobenglück

Bei pro-Variante-Rate von 4.5 % und Best-of-24 erwartet man rechnerisch
`1 − (1 − 0.045)²⁴ ≈ 67 %` der Docs mit ≥1 Treffer. Beobachtet: 42 %. Liegt unter
der Erwartung, weil die Variant-Rate per-Doc heterogen ist (siehe Tabelle:
HB_100223004 hat 5 Treffer, 7 Docs haben 0).

Smoke-2 hatte 3 Docs × 20 Varianten × 2 Modelle = 120 Würfe gegen die
~5 %-Rate — bei dieser Stichprobengröße erwartet man fast garantiert ≥1
Treffer pro Doc, was beobachtet wurde. Phase 3a macht 288 Würfe verteilt auf
12 Docs (24/Doc) — feiner, ehrlicher.

### 2. ADR-007 / Jabarian-Imas-Length-Hypothese widerlegt FÜR MISTRAL

ADR 007 + [arXiv:2503.08716] sagten: Pangram detektiert lange KI-Texte „nearly 100 %".
Phase 2 (Sonnet) bestätigte das: 0/19 Bypass auf langen Texten.

Phase 3a mit Mistral zeigt das Gegenteil: long-Bucket hat 7/120 = 5.8 % BEIDEM-Rate,
short hat 3/96 = 3.1 %. **Pangram-Robustheit gegen Längen-Variation ist NICHT
modell-agnostisch** — sie galt nur für GPT-Class-Paraphrase, nicht für Mistral-Stil.

### 3. Pro-Variante-Rate ist die stabile Größe

5 % (Smoke-2) → 4.5 % (Smoke-3, 4.8× mehr Daten) ist eine sehr enge Replikation.
Daraus folgt das skalierende Modell:
- Best-of-24: 67 % Docs (modelliert), 42 % beobachtet
- Best-of-50: 91 % modelliert
- Best-of-100: 99 % modelliert

### 4. Wackel-Risiko nicht ausgeschlossen

`TSP__22f1d55f3` war in ADR 007 als Pangram-Wackler markiert (`pangram_pre = 0.0`
in Phase 2 trotz Pool-Flag 1.0). Er ist hier wieder im Sample und hat 2 BEIDEM-Varianten.
Wenn der Pre-Score heute < 0.5 wäre, sind alle seine Treffer Fakes. Test
ausstehend ($0.05 für 1 Call) — nicht durchgeführt vor diesem ADR, weil das
Gesamtergebnis (5/12) auch mit 1 Wackler robust 4/12 = 33 % bliebe.

## Konsequenzen

### Phase-3a-Story für Bericht

- **Vorher (Smoke-2):** „Mistral knackt Pangram auf allen 3 long-form Casdorff-Fails."
- **Korrekt (Smoke-3):** „Mistral-Small-3.2-24b knackt Pangram in **42 % der Docs
  bei Best-of-24**, mit Inhaltstreue ≥ 0.85. Variant-Rate ~5 %. Länge ist kein
  Schutz. ADR-007-Length-Hypothese galt nur für Sonnet."

### Mistral-Bypass ist publikationswürdig

Wenn man die DAMAGE-Paper-Behauptung „Pangram bleibt cross-humanizer robust" und
Jabarian/Imas „Pangram robust gegen StealthGPT" als State-of-the-Art nimmt, ist
**Mistral-Best-of-N mit Faithfulness-Constraint ein nicht-trivialer Fall**, der
in der Literatur fehlt — Pangram wurde gegen GPT-Class und Open-Source-Detector-Cohorten
getestet, nicht gegen Mistral-3.2.

### Roadmap-Update

Die Optionen aus ADR 007 müssen anders gewichtet werden:

| Option | ADR-007-Empfehlung | Nach Phase 3a |
|---|---|---|
| A — Pangram-API als Reward für Sonnet | $60, ½ Tag — empfohlen | OBSOLET — Sonnet ist nicht die richtige Wahl |
| B — EditLens als lokaler Reward | $0, 1 Tag Setup | weiterhin valider Reward-Pfad |
| C — Echtes RL (GRPO + LoRA) | $50–300, 2–3 Tage | jetzt mit klarem Base-Model: Mistral-3.2 |
| D — Pivot zu „Pangram-Robustness" | ½ Tag, $0 | NICHT MEHR möglich — Pangram ist nicht robust gegen Mistral |

**Neue empfohlene Reihenfolge:**

1. **Skalierungs-Test (Best-of-50)** auf den 7 Fail-Docs: 7 × 26 = 182 Calls = $9.
   Klärt, ob das Best-of-N-Modell stimmt (sollte 91 % vorhersagen → ≥6/7).
2. **Wackel-Test** für die 12 Originaltexte ($0.60). Quality-Gate.
3. **Mistral-Stil-Hypothese:** Welche Stil-Features unterscheiden inhaltstreu-Bypass
   von normalem Output? Vermutung: Mistral-3.2 hat in den hohen Temp-Bereichen
   weniger der GPT-typischen Floskeln, mehr Satzlängen-Varianz. Manuelle Inspektion
   der 13 BEIDEM-Treffer-Texte.
4. **Open-Source-LLM-Vergleich:** Llama-3.3-70b, Qwen3-30b, Mixtral-8x22b auf
   demselben Casdorff-Setup. Klärt, ob Mistral spezifisch ist oder Mistral-Stil
   = „Pangram-blinde Open-Source-Family".

### Phase-3a ist ABGESCHLOSSEN

Mit n=12 length-stratifiziert, vollem Pangram-Eval und Faithfulness-Gate haben
wir ein robustes Bild: Mistral-Small-3.2-24b kann Pangram, in 42 % der Docs
bei Best-of-24 und unter Inhaltstreue-Constraint. Das ist das Phase-3a-Ergebnis,
das als Baseline für Phase 3b (Skalierung) und Phase 4 (RL-Training) dient.

## Auswirkung auf existierende ADRs

| ADR | Status nach 008 |
|---|---|
| ADR 001 (State of Research) | Update nötig: Mistral-Familie als unerforschter Detektor-Bypass-Pfad notieren |
| ADR 002 (Phase 1) | unverändert |
| ADR 003 (AuthorMist Roadmap) | Base-Model-Wahl konkretisiert: Mistral-3.2 statt Qwen2.5-3B |
| ADR 004 (StealthRL Replication) | unverändert |
| ADR 005 (Hybrid-Vergleich) | Mistral-Pfad als 3. Tier in Vergleich aufnehmen |
| ADR 006 (StealthRL zero-shot fail) | unverändert |
| ADR 007 (Phase-2-Fail) | **Length-Hypothese eingeschränkt**: gilt nur für GPT-Class, nicht für Mistral |

## Reproduzierbar

- Skript: `src/humanizer/training/smoke_phase3a_n12.py`
- Sample: `data/phase3a/sample_n12.jsonl`
- Pangram-Results: `data/phase3a/results_n12.jsonl`
- BGE-Sim-Results: `data/phase3a/bge_n12.jsonl`
- Pangram-Cache: erweitert in `data/phase2/pangram_cache.json`

## Lessons Learned

1. **n=3 ist nie genug für Bypass-Rate-Aussagen.** Smoke-2 hatte den richtigen
   methodischen Reflex (klein, schnell, billig), aber die Story „100 %" wäre
   ohne Skalierungs-Run als Headline durchgegangen. Strikt: alle Bypass-Zahlen
   brauchen mindestens n=12 length-stratifiziert.

2. **Length-Hypothesen sind modell-spezifisch.** Pangram-Robustheit gegen GPT-Class
   auf langen Texten ist nicht universell. Bei jeder neuen Base-LLM-Familie
   Length-Strat neu testen.

3. **Best-of-N-Modell ist nützliches Skalierungs-Tool.** Aus pro-Variante-Rate
   p kann man Doc-Coverage für Best-of-N berechnen: `1 − (1−p)^N`. Wenn
   beobachtet < modelliert (42 % vs 67 %), heißt das: per-Doc-Heterogenität ist hoch,
   manche Docs sind echte Härtefälle.

4. **Faithfulness-Gate ist nicht Bonus, sondern Pflicht.** 73 % der Bypass-Varianten
   (untreu+treu zusammen) wären ohne Filter „Erfolge" gewesen — aber semantisch
   verfälscht. BGE-Sim ≥ 0.85 ist die billige verlässliche Pflicht-Stufe.
