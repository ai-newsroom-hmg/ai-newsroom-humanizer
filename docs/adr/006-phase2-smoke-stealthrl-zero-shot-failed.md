# ADR 006 — Phase-2-Smoke: StealthRL zero-shot auf Deutsch versagt

**Status:** Akzeptiert (negativer Befund)
**Datum:** 2026-06-15
**Kontext:** Erster Smoke-Test der Phase-2-Roadmap. Belegt empirisch, dass der StealthRL-Paper-
Checkpoint OHNE Sprach-Transfer-Fine-Tune auf deutsche Tagesspiegel-Texte nicht funktioniert.

---

## Setup

- **Korpus:** 5 Casdorff-Held-out-Artikel aus `data/phase2-training-pool/eval.jsonl`
  - 2 Politik-Kommentare + 3 kürzere Service/Kultur-Artikel
  - Alle mit Pangram `fraction_ai = 1.0` (vor-test, in `ki-check.db`)
- **Paraphraser:** StealthRL-Adapter `suraj-ranganath/StealthRL` auf `Qwen/Qwen3-4B` Base
  via PEFT auf ruediger (M5 Max 128 GB, MLX-Stack)
- **Detector:** Pangram 3.2 API (kommerziell, Tier 3 lt. ADR 005)
- **Pre-Smoke-Verifikation:**
  - `chatgpt-detector-roberta` (Hello-SimpleAI): unbrauchbar auf deutsch (alle 0.0)
  - `MAGE` (yaful/MAGE): transformers-5.9-Schema-Konflikt beim Load
  - EditLens: gated, ohne Token nicht verfügbar (ADR 005 Tier 1+2 offen)

## Resultate

```
Artikel                                      | Pangram pre | Pangram post
2026-02-05 Technische Universität            | 1.0         | 1.0
2026-05-16 ESC-Kommentar                     | 1.0         | 1.0
2026-01-21 Zugunglück in Spanien             | 1.0         | 1.0
2026-02-10 Neuverfilmung                     | 1.0         | 1.0
2026-01-09 Literaturpreis                    | 1.0         | 1.0

Bypass-Rate: 0/5 = 0 %
Pangram-Kosten: ~$0.10 (10 Calls)
```

Paraphrase-Latenz: ~60 s pro Artikel auf ruediger.

## Qualitative Beobachtungen am Output

- StealthRL behält Tagesspiegel-Headers + Headlines bei („Tagesspiegel vom 05.02.2026, Seite B22 / Wissenschaft in Berlin")
- Wiederholt Titel mehrfach im Output („Wissenschaft in Berlin / Technische Universität")
- Inhaltliche Umformung minimal — eher Surface-Edits als deep paraphrase
- Output-Länge variabel (kürzt 4936 → 2948 chars beim ESC-Kommentar)

## Diagnose

1. **Sprach-Mismatch:** StealthRL ist auf MAGE-Pool trainiert (überwiegend englisch). Qwen3-4B-Base
   versteht Deutsch zwar, aber die LoRA-Adapter-Policy ist auf englische Reward-Signale gelernt.
2. **Out-of-distribution Bypass-Policy:** Die im Paper publizierten 97,6 % gelten für RoBERTa,
   FastDetectGPT, Binoculars, MAGE — alles englisch-trainierte Detectors. Pangram (cross-lingual,
   multi-sprachig) ist eine andere Detector-Klasse.
3. **Adapter-Output-Format:** Behält Metadaten — Hinweis, dass das Modell den Original-Text nicht
   "internalisiert" und neu generiert, sondern oberflächlich umformt.

## Konsequenz für die Roadmap

| Pfad | Status |
|---|---|
| ADR 003 (AuthorMist eigenes Training mit EditLens lokal als Reward) | weiterhin valider Pfad — needs Fine-Tune-Cycle |
| ADR 004 Schritt 2 (StealthRL zero-shot) | **WIDERLEGT** für Deutsch — durchgeführt 2026-06-15 |
| ADR 004 Schritt 3 (StealthRL Fine-Tune auf Casdorff-Train-Pool) | nächster sinnvoller Schritt — braucht HF-Token + EditLens-Zugang |
| ADR 005 (Hybrid-Vergleich) | Methode bleibt — Daten sind robuster wenn ADR 003 + ADR 004 Schritt 3 beide gelaufen sind |

## Reproduzierbar

- Code: `src/humanizer/training/stealthrl_smoke.py`
- Score-Skript: `src/humanizer/training/score_paraphrases.py`
- Smoke-Output: `data/phase2/smoke/stealthrl_editlens.jsonl`
- Log: `data/phase2/smoke/run.log`

## Wissenschaftlicher Beitrag dieses Befunds

Empirisch belegte Sprach-Transfer-Grenze für RL-detector-aware Bypass-Methoden. Bestätigt
Pangrams cross-language-Robustheit auf deutschsprachiger Politik-Kommentar-Domäne. Bisher
nicht publiziert; ein klares Datenpunkt für die in ADR 005 vorgesehene Vergleichs-Studie.
