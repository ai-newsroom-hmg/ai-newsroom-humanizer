# ADR 001 — State of Research zu AI-Humanizern und Pangram-Robustheit

**Status:** Akzeptiert
**Datum:** 2026-06-15
**Kontext:** Auftrag, einen Humanizer zu bauen, der Pangram austrickst — ohne Inhalt zu verfälschen.

---

## Entscheidungsbasis

### Was die Wissenschaft 2025–2026 sagt

| Pfad | Paper | Methode | Erfolg | Pangram getestet? |
|---|---|---|---|---|
| **DIPPER** | Krishna et al., NeurIPS 2023 (arXiv:2303.13408) | T5-11B Paraphraser, L/O-Parameter | DetectGPT 70 % → 4,6 % | **Ja, hält** |
| **Recursive Paraphrasing** | Sadasivan et al., ICML 2024 (arXiv:2303.11156) | Selber Text n-mal durch Paraphraser | Watermark 99 % → 9,7 % | **Nicht publiziert** |
| **RADAR** | Hu, Chen, Ho, NeurIPS 2023 (arXiv:2307.03838) | Adversarial Co-Training Paraphraser ↔ Detektor | high | Nein |
| **SICO** | Lu et al., TMLR 2024 (arXiv:2305.10847) | In-Context Few-Shot detektor-resistent | umgeht GPTZero, OpenAI-Classifier | Nein |
| **AuthorMist** | David & Gervais, ETH Zürich 2025 (arXiv:2503.08716) | Qwen2.5-3B + GRPO mit Detektor-API als Reward | **78,6–96,2 %** auf GPTZero, WinstonAI, Originality, Sapling | **Pangram NICHT im Eval-Set** |
| **AuthorMix** | Saarland 2026-03 (arXiv:2603.23069) | Per-Autor-LoRA + Layer-Gating | schlägt GPT-5.1 für low-resource | Nein |
| **SpaceInfi** | arXiv 2307.02599 | Whitespace vor Komma | senkt GPTZero deutlich | Nein |
| **Unicode-Homoglyph / Light-Edit** | Creo et al. 2024 | Triviale Edits | 6 Detektoren 39,5 % → 17,4 % | Marketing: hält |
| **DAMAGE** (Defense) | Pangram Labs 2025 (arXiv:2501.03437) | Pangram greift sich selbst mit fine-tune an | Selbstangabe: robust | (Pangrams eigene Studie) |
| **„Base Models Look Human"** | 2026 (arXiv:2605.19516) | Welche Base-LLMs umgehen Detektion ohne Training? | Verschiebt das Problem | — |

### Pangrams öffentlicher Stand

- **Eigene Marketing-Behauptung (August-2025-Blog):** 97 % Recall auf humanisiertem Text gegen 19 kommerzielle Tools (StealthGPT, Undetectable AI, QuillBot, Phrasly, HIX Bypass, BypassGPT, Stealth Writer, EvadeGPT, Humaniser, Smodin, HumanizeAI.com, Humbot etc.)
- **Selbsteinschätzung (DAMAGE-Paper §7):** *„Future work — increase robustness to detector-aware adversarial attacks"* — Hersteller räumt offene Flanke ein.
- **Unabhängiger Audit (Jabarian & Imas 2025, [BFI WP 2025-116](https://bfi.uchicago.edu/wp-content/uploads/2025/09/BFI_WP_2025-116.pdf), UChicago):** FNR ≈ 0 % auf langen Texten, ~5 % auf kurzen. Nur StealthGPT-Default-Rewrite getestet.

### Was bedeutet das für unseren Stack

Pangram ist *gegen alle öffentlich getesteten Klassen von Paraphrase-Humanizern* gehärtet.
Die **drei strukturell offenen Flanken** der Literatur sind:

1. **AuthorMist-Klasse**: RL-Training mit Pangram als Reward (nicht publiziert).
2. **AuthorMix-Klasse**: Per-Autor-LoRA aus echtem menschlichem Korpus (nicht publiziert gegen Pangram).
3. **Hybrid-Edit / Source-Confusion** (Zindulka 2024): KI-Text mit echten Menschen-Sätzen mischen — funktioniert (siehe ADR 002), aber **verletzt Inhaltstreue**.

---

## Konsequenz

→ Phase 1 baut Pfade die **OHNE Training** auskommen (Prompt-Engineering, Sentence-Mix).
   Ziel: Nachweis ob es ohne RL geht. Ergebnis: nein (siehe ADR 002).

→ Phase 2 (Roadmap) implementiert AuthorMist/AuthorMix mit GRPO und Pangram-Reward.
   Erwartet die wissenschaftliche Lücke zu schließen.

→ Dual-Use-Reflexion: Resultate veröffentlichungswürdig als „Independent Audit of Pangram Robustness
   vs Detector-Aware Adversarial Attacks (2026)". Spannung zwischen Plagiats-Hilfe und Detektor-Härtung
   anerkannt — Forschung dient Härtung, nicht Anleitung.

---

## Quellen

Siehe `README.md` Section „Wissenschaftlicher Stand 2026" + Vault-Notes:

- `🧠 Arbeiten/Notizen/AI-Detektoren – Wissenschaftliche Paper Close-Reading 2026-06-13.md`
- `🧠 Arbeiten/Notizen/KI-Autorschafts-Verwischung – Source Confusion bei Mensch-KI-Co-Schreiben 2026-06-13.md`
- `🧠 Arbeiten/Notizen/Close Reading – Emi  Spero 2024  Technical Report on the Pangram AI-Genera.md`
- `_skills/journalism/style-transfer-2026-methodology-survey.md`
- `_skills/journalism/authormix-implementation-pattern.md`
