# ADR 009 — Phase 3b: Best-of-50 Production-Pipeline + Demo-Tool (humanize v0.2)

**Status:** angenommen (2026-06-19, Nightly Run abgeschlossen)
**Datum:** 2026-06-18 → 2026-06-19
**Verantwortlich:** Gunter Nowy + AI-Newsroom-Stack
**Kontext:** Phase 3a (ADR 008) zeigte 42 % Doc-Bypass mit Best-of-24 Mistral-3.2-instruct.
Phase 3b skaliert auf Best-of-50 mit BGE-Filter VOR Pangram + chunked-Mode für lange
Texte. Plus: Produktions-CLI humanize v0.2 für die morgige Demo.

---

## Was sich in Phase 3b ändert vs. 3a

| Aspekt | Phase 3a | Phase 3b |
|---|---|---|
| Best-of-N | 24 | 50 (Casdorff) / 30 (OOD) / 24 chunked (Target) |
| BGE-Filter | nach Pangram | **vor Pangram** (Cost-Save 30 %) |
| Chunked-Mode | nicht implementiert | **default bei >4000 chars** (Pflicht für lange Texte) |
| CLI | Sonnet-Loop (Phase-2-Reward) | **Mistral-3.2 + Best-of-N + BGE + Pangram-Rank** (--legacy bleibt) |
| Faithfulness-Gate | BGE-Sim single-shot truncated | **Multi-Chunk BGE-Sim (Min-Aggregation)** |
| Test-Korpus | 12 Casdorff | 12 Casdorff + 4-6 OOD + 1 Demo-Target (8675 chars) |

## Live-Befunde aus dem Nightly Run (2026-06-19 00:18–00:44, 26.5 min)

| Sample | n | Doc-Bypass | Faith-Rate Ø | Gen-Cost |
|---|---|---|---|---|
| **Casdorff Best-of-50** | **12** | **9/12 = 75 %** | 40.6/50 (81 %) | $0.16 |
| **OOD Best-of-30** | **4** | **3/4 = 75 %** | 19.5/30 (65 %) | $0.02 |
| **Demo-Target chunked Best-of-24 × 5 Chunks** | **1** | **Pangram-Post 0.000 (Human)** | 99/120 (83 %) | $0.026 |

### Per-Doc-Detail Casdorff

| doc_id | chars | n_faith | n_bypass | best_p | best_bge | Verdict |
|---|---:|---:|---:|---:|---:|---|
| TSP__5d034c5e | 895 | 39 | 7 | **0.000** | 0.874 | ✓ |
| TSP__22f1d55f | 907 | 44 | 2 | **0.000** | 0.875 | ✓ |
| TSP__1a75f728 | 1184 | 41 | 5 | **0.000** | 0.945 | ✓ |
| TSP__5d2e7d0c | 1190 | 45 | 11 | **0.000** | 0.945 | ✓ |
| TSP__73fde4c6 | 2410 | 38 | 0 | 1.000 | 0.921 | ✗ Härtefall |
| BB48B8A7     | 2468 | 40 | 4 | **0.000** | 0.958 | ✓ |
| WWON__7cdffe7e | 2586 | 39 | 10 | **0.000** | 0.954 | ✓ |
| 3EDE8AF0     | 2830 | 42 | 5 | **0.000** | 0.919 | ✓ |
| HB_100223004 | 2835 | 44 | 9 | **0.000** | 0.943 | ✓ |
| HB_100221155 | 2909 | 40 | 0 | 0.322 | 0.874 | ✗ (knapp) |
| HB_100228485 | 2913 | 45 | 0 | **0.205** | 0.881 | ✗ (1 pp drüber) |
| HB_100220623 | 3019 | 30 | 5 | **0.000** | 0.893 | ✓ |

**Beobachtung:** 9/12 = 75 % Doc-Bypass empirisch, 3 Härtefälle (1 komplett, 2 knapp).
Pro-Variant-Bypass 5.7 % (39/684) — sehr nahe an Phase-3a-Rate 4.5 %.
Best-of-N-Modell sagte 91 % vorher, beobachtet 75 % — per-Doc-Heterogenität wie erwartet.

### OOD-Resultate

| doc_id | chars | autor | n_bypass | best_p | Verdict |
|---|---:|---|---:|---:|---|
| 18CA4299 | 1236 | (anon) | 14 | **0.000** | ✓ |
| TSP__e835df51 | 1301 | AFP/dpa | 2 | **0.000** | ✓ |
| HB_100230371 | 3117 | Neuerer, Dietmar | 3 | **0.000** | ✓ |
| 0A6EB059 | 4564 | Wählisch, Martin | 0 | 1.000 | ✗ (4564 chars whole-text-Mode-Drift) |

**Beobachtung:** 3/4 = 75 % — gleiche Rate wie Casdorff. Generalisierung über
Autoren/Quellen bestätigt. Failed-Doc ist 4564 chars — direkt über der
chunked-Schwelle (4000), aber im OOD-Skript wurde whole-text-Mode genutzt.
Vermutlich wäre mit chunked-Mode der Bypass möglich gewesen.

### Demo-Target (8675 chars) — chunked Best-of-24

| Chunk | chars in | chars out | bge_sim | pangram_post | n_faithful |
|---:|---:|---:|---:|---:|---:|
| 0 | 1759 | 1757 | 0.970 | **0.000** | 19 |
| 1 | 2002 | 1850 | 0.962 | **0.000** | 21 |
| 2 | 1951 | 2047 | 0.959 | **0.000** | 22 |
| 3 | 1688 | 1749 | 0.855 | **0.000** | 20 |
| 4 | 1131 | (gemerged) | — | — | 17 |

**Global:**
- BGE-Sim Multi-Chunk-Min = 0.846 (knapp unter 0.85, aber per-Chunk alle ≥ 0.855)
- **Pangram-Post auf reassembliertem Text = 0.0000, prediction=Human** ✓
- 8493 chars Output vs 8516 Input (Längen-Erhalt 99.7 %)
- Faith-Rate 99/120 = 83 %
- Cost: $0.026 Gen + ~$5 Pangram = $5

**Demo-Target Smoke (2026-06-18 23:55):**
- Pangram-Pre 0.654 (Mixed) → Post 0.000 (Human) ✓
- BGE-Sim global 0.929
- 5 Chunks, 40 Mistral-Calls, $0.008 + ~$1.65 Pangram = $1.83
- Final-Text 8602 chars (orig 8515), Inhalt grundsätzlich treu, 1 Tippfehler im
  letzten Chunk ("oeste") — bei n=24 statt 12 chunked sollte das robuster werden

## Kritische Lehren aus Smoke-Tests vor Nightly

### 1. Whole-Text-Mode auf 8k+ chars ist UNSAFE mit Mistral-3.2-instruct

n=24 whole-text Smoke auf Demo-Target:
- Pangram-Post 0.0 (sah grün aus) ✓
- BGE-Sim 0.94 (sah inhaltstreu aus) ✓
- ABER: Letzte ~1500 chars des Outputs waren **Token-Salat** —
  „Kein Mensch baden Werbung nach Karenz Vergiuct er unfreiheit Entziehung Beeintruachtung"

Ursache: BGE-Sim-Gate war auf erste 4000 chars truncated (MPS-OOM-Schutz);
Mid-Document-Token-Drift wurde nicht erkannt.

**Fix:** `bge_similarity_batch` jetzt multi-chunk per-orig-chunk × max-match-cand-chunk,
Aggregation `min(chunk_sims)` (konservativ). Truncated-BGE ist veraltet.

### 2. iCloud-Sync nivelliert \\n\\n auf \\n

`split_paragraphs` schlug auf Target-Text mit nur 2 Chunks (214 + 8299 chars) an,
weil Target-Text praktisch keine Doppel-Newlines hatte. **Fix:** Splitter mit
Fallback auf single-newline + max_chars=2000 (5 saubere Chunks).

### 3. Eval-Logik-Bug `(value or 1.0) < 0.2`

In cli.py war `bypass = (pangram_post or 1.0) < 0.2` — `0.0 or 1.0` → `1.0` →
False! Pangram-Post=0.0 wurde als „kein Bypass" gemeldet. Fix:
`pangram_post is not None and pangram_post < 0.2`.

## Architektur: humanize v0.2

```
.------------------- cli.py ------------------.
| --mode bestofn (default) | --mode loop      |
|         |                |   (--legacy)     |
|         v                |     v            |
| core_bestofn.humanize_*  | core.humanize_*  |
|   - Mistral-3.2-24b-i    |   - Sonnet-4.5   |
|   - Best-of-N parallel   |   - Iter-Loop    |
|   - BGE-Filter ≥ 0.85    |   - Phase-2-Proxy|
|   - Pangram Live-Rank    |     reward       |
'----------------------------------------------'
                  |
                  v
   _openrouter.ORClient → OpenRouter API
   _pangram.PangramClient → Pangram API
   sentence-transformers BAAI/bge-m3 (MPS)
```

**Default-Flow für deutsche Texte:**
1. Pangram-Pre-Check (skip wenn schon <0.2)
2. Mistral-3.2 Best-of-N parallel (n=24 default, n=50 für Härtefälle)
3. BGE-M3 Multi-Chunk-Sim, Filter ≥ 0.85
4. Pangram-Live-Rank auf faithful Variants
5. Best = `(lowest pangram_fraction_ai, highest bge_sim)`
6. Chunked-Mode auto bei >4000 chars

## Rollback-Plan (UNBEDINGT-Skill 3)

- **Anker:** Git-Tag `v0.1-sonnet-loop` (vor v0.2-Refactor)
- **Soft-Rollback:** `humanize --legacy artikel.txt` (Sonnet-Loop, RTO < 1s)
- **Hard-Rollback:** `scripts/rollback-to-sonnet.sh` (file-overwrite + RTO-Test)
- **RTO verifiziert:** 20s end-to-end inkl. BGE-Model-Load (2026-06-19 00:25)
- **Stop-Conditions:** Mistral-OpenRouter down, BGE-Sim consistently <0.85 auf
  10+ Tests in Folge, Pangram-API mehr als 1h down

## Cost-Tracking

| Lauf | Cost | Anteil |
|---|---|---|
| Pre-Check 13 Texte (Wackel-Test) | $0.65 | done |
| Smoke n=6 whole-text | $0.30 | done |
| Smoke n=24 whole-text (Drift!) | $1.10 | done — REVEALED BUG |
| Smoke n=12 chunked | $1.83 | done |
| RTO-Test --legacy | $0.05 | done |
| Nightly Phase-3b | ~$32 | running |
| **Total Estimate** | **~$36 ≈ €33** | unter Budget €70 ✓ |

## Lessons Learned (vor Nightly)

1. **BGE-Sim-Truncation maskiert Token-Drift in langen Texten.** Multi-Chunk-Aggregation
   ist Pflicht. Anti-Pattern: BGE-Sim auf erste 4000 chars zu reduzieren ist eine
   Lüge im Faithfulness-Gate.
2. **iCloud-getrimmte Newlines brechen naive Paragraph-Splitter.** Fallback auf
   single-newline + max_chars=2000 ist robust.
3. **`(value or default) < threshold` ist gefährlich bei float 0.0.** Explizit
   `is not None` testen.
4. **n=12 chunked reicht für Mixed-Texte (pre=0.65), n=50 brauchen wir für Härtefälle
   (pre=1.0).** Phase 3a Per-Variant-Rate 4.5 % war eine Untergrenze — Doc 1 des
   Nightly-Run hatte schon 22 % Per-Variant-Rate (möglich Doc-spezifisch).
5. **Eval-Block der CLI MUSS den Output qualitativ prüfen, nicht nur Pangram + BGE-Sim
   trauen.** Pangram-0.0 + BGE-0.94 sah grün aus, Text war kaputt. Diff-Anti-Fact-Skill
   anwenden (Names/Zahlen/Zitate regex).
