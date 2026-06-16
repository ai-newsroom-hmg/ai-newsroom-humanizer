# ADR 007 — Phase-2-Fail-Mode-Diagnose: Proxy-Reward versagt, „Erfolge" sind Längen-Konfounder

**Status:** angenommen
**Datum:** 2026-06-16
**Verantwortlich:** Gunter Nowy + AI-Newsroom-Stack
**Kontext:** Phase 2 schloss mit 5/24 = 21 % Bypass + 24/24 inhaltstreu auf Casdorff ab (commit 931e5e4). Vor Phase 3 (RL-Training) Fail-Mode-Diagnose der 19 gescheiterten Artikel.

---

## Befund — Phase 2 hat nicht funktioniert wie gedacht

### 1. Der Loop bewegt nichts. Best-of-50 = Best-of-10.

Mittlere Proxy-Score-Trajektorie über alle 24 Artikel:

| Iter | Mean | Std |
|------|------|-----|
| 0 (Baseline) | 0.841 | 0.038 |
| 1 | 0.841 | 0.036 |
| 2 | 0.840 | 0.036 |
| 3 | 0.840 | 0.036 |
| 4 | 0.840 | 0.036 |
| 5 | 0.840 | 0.035 |

Δ proxy_post − proxy_pre über alle 24: **mean = −0.0014, std = 0.0044, min = −0.0099, max = +0.0086.**

Der Loop tut messbar nichts. Die best-of-10-Variante in Iter 1 ist genau so gut wie die best-of-50-Variante nach Iter 5. Das wären $0.20 statt $1 pro Artikel — bei identischem Ergebnis.

### 2. Proxy ist von Pangram entkoppelt.

Alle 5 „Bypass-Erfolge" haben `proxy_post ≥ 0.747` — der Proxy hält sie weiterhin für KI. Pangram sagt das Gegenteil (`pangram_post = 0.0`).

| doc_id | proxy_post | pangram_post |
|---|---|---|
| TSP__e835df51b | 0.784 | 0.000 |
| TSP__1a75f7288 | 0.747 | 0.000 |
| TSP__5d2e7d0cc | 0.805 | 0.000 |
| TSP__22f1d55f3 | 0.812 | 0.000 |
| TSP__b52fcb64d | 0.798 | 0.000 |

Der Proxy „weiß" nicht, was Pangram weiß. val_MAE 0.29 war ehrlich — der Proxy kann Pangram nicht hinreichend approximieren, um als Reward zu funktionieren.

### 3. Die 5 Erfolge sind ein Längen-Konfounder.

| Gruppe | n | Mean chars | Median | Min | Max |
|---|---|---|---|---|---|
| Bypass-Erfolg | 5 | 1.195 | 1.190 | 895 | 1.392 |
| Bypass-Fail | 19 | 3.183 | 2.913 | 895 | 5.674 |

Längen-Verteilungen liegen praktisch nicht aufeinander. Alle 5 Erfolge sind kurz, alle Fails sind lang oder mittel-lang. Alle 5 Erfolge sind **TSP-Artikel** (Tagesspiegel), keine HB-, keine WWON-Artikel.

### 4. Daten-Leak: TSP__22f1d55f3 war kein Bypass

`pangram_pre = 0.0` — Pangram hat den Originaltext bereits als Human klassifiziert, NICHT als KI. Der „Erfolg" ist Eval-Artefakt. Eigentlich nicht in `proxy.jsonl` enthalten sein dürfen (Filter `fraction_ai = 1.0` hat hier nicht gegriffen).

**Bereinigte Bypass-Rate: 4/23 = 17 % — auf ausschließlich kurzen TSP-Artikeln.**

---

## Interpretation — was wirklich passiert ist

1. **Sonnet-Temperatur-Sweep erzeugt random Varianten.** Bei kurzen Artikeln gibt es einen kleinen Prozentsatz, der Pangram-Score zufällig in Richtung Human verschiebt. Das ist nicht Optimierung, das ist Roulette mit positivem Erwartungswert auf einer winzigen Teilmenge.

2. **Der Proxy-Reward war wirkungslos.** Best-of-10 ohne Proxy würde dasselbe Ergebnis liefern. Der Proxy wählt nicht die Pangram-besten Varianten aus — er kann sie nicht erkennen.

3. **Pangram ist robust gegen Sonnet-Paraphrase auf längerem deutschem Kommentarstil.** 19/19 lange Artikel = 0 % Bypass.

4. **Die Phase-2-Architektur (Proxy + Best-of-N + Loop) ist für Pangram-Bypass methodisch ungeeignet** — solange der Proxy nicht reward-korreliert ist. Der ADR 003-Plan (EditLens als lokaler Reward) ist genau die richtige Antwort darauf.

---

## Konsequenzen

### Phase-2-Release-Story muss korrigiert werden

- **Vorher:** „5/24 = 21 % Bypass mit Inhaltstreue."
- **Korrekt:** „4/23 = 17 % Bypass — alle auf kurzen TSP-Artikeln (≤ 1.392 chars). Kein Bypass auf längeren Artikeln. Sonnet-Temperatur-Roulette, kein Proxy-Beitrag."

Das CLI ist trotzdem brauchbar — als **Sonnet-Best-of-N-Generator**, nicht als Pangram-Bypass-Tool. Die Iterations-Schleife kann auf 1 reduziert werden ohne Qualitätsverlust (90 % Kostenersparnis).

### Phase 3 muss anders aussehen

**Verworfen:** Proxy-Reward weiterskalieren. Mehr Daten würden den Proxy nicht magisch reward-korreliert machen — der Fehler ist nicht der Proxy, sondern dass Pangram architektonisch anders denkt als BGE-M3-Embeddings.

**Optionen für Phase 3, sortiert nach Hebel:**

#### A) Pangram-API direkt als Reward — schnellster Realitätscheck

- Bei jeder Iter Pangram live aufrufen, beste Variante per echtem Pangram-Score wählen
- Cost: ~$2.50/Artikel (50 Pangram-Calls × $0.05) — bei Phase-2-Korpus = $60 für vollen Re-Run
- Liefert die echte Obergrenze von Best-of-N-Sonnet-Paraphrase. Wenn das auf langen Artikeln auch 0 % bringt, ist die Architektur nachweislich falsch — RL kann nicht retten, was Sonnet selbst nicht produziert.

#### B) EditLens als lokaler Reward (ADR 003) — der dokumentierte Plan

- HF-Token + License akzeptieren
- EditLens-RoBERTa lokal als Reward → ersetzt Proxy
- Korrelation EditLens ↔ Pangram ist offen (Pangram Labs hat EditLens trainiert, also: sollte hoch korrelieren)
- Aufwand: 1 Tag Setup, gleicher Loop-Code

#### C) Echtes RL mit Pangram-Reward (StealthRL/AuthorMist Schritt 3) — der teure Pfad

- Qwen3-4B + GRPO + LoRA auf Casdorff-Train-Pool
- Reward: EditLens (lokal, $0) oder Pangram-API ($150–300 für Training)
- Erwartung: 60–90 % Bypass (StealthRL-Niveau auf 4 Open-Source-Detektoren, hier offen für Pangram)
- Aufwand: 2–3 Tage ruediger, $50–300

#### D) Pivot — Pangram-Robustness als Research-Outcome

- Das Phase-2-Ergebnis ist publikationswürdig: „Pangram ist robust gegen GPT-Class Best-of-N-Paraphrase auf deutschem Kommentarstil"
- ADR 005 hatte den Vergleich vorgesehen — jetzt mit klarem Negativbefund auf einer Methode
- Aufwand: ½ Tag Schreiben

### Empfohlene Reihenfolge

1. **Option A zuerst** — $60, ½ Tag. Klärt ob Sonnet-Paraphrase überhaupt Pangram knacken kann (Methoden-Decke vs. Modell-Decke).
2. Wenn A positiv: Option B (EditLens-Reward für günstige Iteration)
3. Wenn A negativ: Option C (RL ist notwendig) oder Option D (Pivot)

---

## Auswirkung auf existierende ADRs

| ADR | Status nach 007 |
|---|---|
| ADR 002 (Phase 1) | unverändert — Vergleichsbasis |
| ADR 003 (AuthorMist mit EditLens) | weiterhin valider Pfad — jetzt einer von mehreren |
| ADR 004 Schritt 3 (StealthRL Fine-Tune) | weiterhin valider Pfad |
| ADR 005 (Hybrid-Vergleich) | **erweitert** — Phase-2-Proxy-Approach ist jetzt ein **Negativbefund** in der Vergleichstabelle |
| ADR 006 (StealthRL zero-shot fail) | unverändert |

## Reproduzierbar

- Eval-Daten: `data/phase2/eval_results.jsonl`, `data/phase2/loop_results.jsonl`
- Diagnose-Skript: siehe Bash-Block in git-Historie zu commit 931e5e4 + Folge-Commit dieses ADRs
- Schlüssel-Datenpunkte oben aus 24 × 1 Artikel × 6 Iter = 144 Proxy-Score-Messungen

## Lessons Learned

1. **Reward-Korrelation muss VOR Loop-Build validiert werden.** Wir haben den Proxy gegen Pangram-Labels trainiert (MSE), aber nie geprüft, ob er auf Sonnet-Paraphrase-Varianten dasselbe sagt wie Pangram. Das ist die eigentliche Frage für einen Reward.

2. **Konfounder „Doc-Länge" gehört in jeden Bypass-Plot.** Ohne die Längen-Stratifikation hätte das 21 %-Ergebnis falsch verallgemeinert.

3. **Trajektorie-Plots sind günstig und decken systematische Probleme auf.** 30 Minuten Analyse → 1 ADR, das das Roadmap-Budget für Phase 3 von ~$300 auf zielgerichtete Tests reduziert.
