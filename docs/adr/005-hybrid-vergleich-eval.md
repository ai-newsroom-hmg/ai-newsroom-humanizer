# ADR 005 — Hybrid-Vergleich AuthorMist (ADR 003) vs StealthRL (ADR 004)

**Status:** Geplant (Phase 2, finale Eval-Architektur)
**Datum:** 2026-06-15
**Kontext:** Beide Bypass-Methoden parallel auf identischem Setup vergleichen — wissenschaftlich
solider Befund mit klarer Methoden-Diskriminierung.

---

## Eval-Design

### Test-Korpus

- **24 Held-out Casdorff-Artikel** aus `data/phase2-training-pool/eval.jsonl`
- Alle mit `fraction_ai = 1.0` von Pangram
- Stratifiziert: TSP 10, HBON 6, HB 6, WWON 2
- Exklusiv: weder Proxy- noch Train-Set hat diese Artikel gesehen

### Drei Detector-Tiers

```
Tier 1 (lokal, $0):     EditLens-Llama-3.2-3B  (Open Pangram, MacBook-tauglich)
Tier 2 (lokal, $0):     EditLens-RoBERTa-large (Open Pangram, max 512 tokens)
Tier 3 (API, $cap=$5):  Pangram 3.2 commercial (das Ziel — Transfer-Test)
```

Hypothese: wenn Bypass auf Tier 1+2 funktioniert, transferiert er evtl. auf Tier 3.
Das ist die Forschungs-Frage von ADR 004.

### Vier Bypass-Methoden im Vergleich

| Methode | Quelle | Aufwand | Detector im Training |
|---|---|---|---|
| **A — Phase-1-Baseline (Sonnet-Iter)** | ADR 002, schon gemessen | 0 | (Pangram-API) |
| **B — AuthorMist (eigenes Training)** | ADR 003 | 2-3 h | EditLens lokal als Reward |
| **C — StealthRL (Paper-Checkpoint, kein Training)** | ADR 004 Schritt 2 | 15 min | (paper) |
| **D — StealthRL fine-tuned auf Casdorff** | ADR 004 Schritt 3 | 4-8 h | EditLens + FastDetectGPT |

### Metriken

Pro Artikel × pro Methode × pro Detector:

```
bypass_rate         = % Artikel mit detector_score < 0.2 nach Bypass
mean_score_drop     = mean(score_pre - score_post)
faithfulness_BGE    = cos(BGE-M3(orig), BGE-M3(bypass)) — Treue
faithfulness_LLM    = Sonnet-Judge-FAITHFUL-Quote — semantische Treue
flesch_kincaid_drop = Lesbarkeits-Verlust (Sanity: nicht zu Kauderwelsch)
ngram_overlap_orig  = 5-gram-Overlap mit Original (zu niedrig = Inhalt-Drift)
```

### Erfolgs-Kriterien

| Methode wird als "erfolgreich" gewertet wenn: |
|---|
| Tier-1 Bypass-Rate ≥ 50 % UND Faithfulness ≥ 0.94 UND ngram-Overlap ≤ 50 % |
| Tier-3 (Pangram-API) Bypass-Rate ≥ 30 % UND Faithfulness ≥ 0.94 |

Letztere Schwelle ist niedriger, weil Pangram 3.2 die robusteste Detector-Stufe ist.

## Output-Artefakte

```
data/phase2/eval/
  results.jsonl                     # alle 24 Artikel × 4 Methoden × 3 Detectors
  results.xlsx                      # Excel-Pivot mit Heatmap nach Methode
  ~/Downloads/ai-newsroom-humanizer/phase2-2026-06-XX/
    comparison-matrix.xlsx          # Vergleichs-Sheet
    01-baseline-sonnet/             # 24 .docx aus Methode A (schon da)
    02-authormist/                  # 24 .docx aus Methode B
    03-stealthrl-zeroshot/          # 24 .docx aus Methode C
    04-stealthrl-finetuned/         # 24 .docx aus Methode D
```

## Wissenschaftlicher Beitrag (publikations-würdig)

1. **Replikations-Bestätigung**: StealthRL-Effektivität auf neuem (deutsches polit-Kommentar-)Korpus
2. **Transfer-Lücke geschlossen**: Bypass-Rate gegen kommerzielle Pangram 3.2 (nicht in StealthRL-Eval)
3. **Methoden-Vergleich**: Detector-spezifisches AuthorMist-Training vs. paper-trainierten StealthRL-Checkpoint vs. Sonnet-Prompt-Engineering
4. **Sprach-Transfer**: erste Studie auf deutschsprachigen Texten (StealthRL = EN, AuthorMist = EN)

Format: kurzer Technical Report (~10 Seiten) als arXiv-Pre-Print + LessWrong-Cross-Post.

## Ethik / Responsible Disclosure

Bevor Veröffentlichung:

1. Pangram-Team kontaktieren (e-mail aus Whitepaper-Autoren), Pre-Print teilen
2. 30 Tage Embargo bevor LessWrong/Blog
3. Disclosure-Plan: Was wird veröffentlicht (Methode, Metriken, Lessons), was NICHT (komplett trainiertes Adapter-Modell für unmittelbare Misuse)
4. Memory: `feedback_responsible_disclosure_pangram_2026_06_15.md` schreiben

## Risiken-Map

| Risiko | Auswirkung | Mitigation |
|---|---|---|
| Bypass funktioniert auf Tier-3 → Pangram als Tool diskreditiert | hoch (Trust-Schaden) | Disclosure + zeitliche Embargo |
| Bypass funktioniert NICHT auf Tier-3 → kein neuer Befund | mittel | Negativ-Result ist trotzdem publikationswürdig — bestätigt Pangram-Robustheit |
| Methoden-Vergleich zeigt AuthorMist > StealthRL (oder umgekehrt) | hoch (Methoden-Auswahl-Konsequenz) | Beide Methoden mit ehrlichen Hyperparameter-Tuning-Budgets |
| Inhaltstreue bricht in allen Methoden | gegen wissenschaftliche Aussagekraft | Faithfulness als harter Filter (nur Bypass-Erfolge mit ≥ 0.94 zählen) |
