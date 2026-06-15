"""Phase-2-Smoke: StealthRL-Bypass + EditLens-Detector + MAGE auf 5 Casdorff-Artikeln.

Was es macht:
  1. Lade 5 Casdorff-Held-out-Artikel aus data/phase2-training-pool/eval.jsonl
  2. Lade EditLens-RoBERTa-large (Open Pangram) lokal -> pre-Score je Artikel
  3. Lade StealthRL-Modell (HF: suraj-ranganath/StealthRL) als PEFT-Adapter auf Qwen3-4B-Base
  4. Paraphrasiere jeden Artikel mit StealthRL
  5. Score Paraphrasen mit EditLens + MAGE
  6. Speichere Pre/Post-Vergleich als JSONL

Output: data/phase2/smoke/stealthrl_editlens.jsonl

Voraussetzungen auf ruediger:
  - ~/.cache/huggingface/token vorhanden (mit Lizenz-Zustimmungen fuer EditLens)
  - mlx-lm-lora 2.1.0, transformers 5.9, sentence-transformers 5.5.1 (alle da)
  - Qwen3-4B-Base wird automatisch von HF gezogen (~8 GB, einmalig)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def load_eval_articles(limit: int = 5) -> list[dict]:
    fp = Path(__file__).resolve().parents[3] / "data" / "phase2-training-pool" / "eval.jsonl"
    if not fp.exists():
        sys.exit(f"eval-pool fehlt: {fp}")
    rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit]


def setup_editlens_detector():
    """EditLens-RoBERTa-large lokal als AI/Human-Classifier laden."""
    print("[detector] EditLens-RoBERTa-large laden...", flush=True)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_id = "pangram/editlens_roberta-large"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    def score(text: str) -> dict:
        with torch.no_grad():
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        # EditLens-Label-Reihenfolge gemaess Modell-Card pruefen!
        # Annahme: [human, ai_assisted, ai] oder [human, ai] — wir reporten alle Klassen
        return {"probs": probs, "argmax_idx": int(torch.argmax(logits, dim=-1))}

    print(f"[detector] EditLens-RoBERTa geladen, classes: {model.config.num_labels}", flush=True)
    return score


def setup_mage_detector():
    """MAGE-Detector als Fallback (publik verfuegbar, kein Token)."""
    print("[detector] MAGE als Fallback laden...", flush=True)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_id = "yaful/MAGE"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    def score(text: str) -> dict:
        with torch.no_grad():
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        return {"probs": probs, "argmax_idx": int(torch.argmax(logits, dim=-1))}

    return score


def setup_stealthrl_paraphraser():
    """StealthRL als PEFT-Adapter auf Qwen3-4B (oder welches Base-Modell laut Modell-Card)."""
    print("[paraphraser] StealthRL-Adapter laden...", flush=True)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Base-Modell laut StealthRL-Paper: Qwen3-4B
    base_id = "Qwen/Qwen3-4B"
    adapter_id = "suraj-ranganath/StealthRL"

    tok = AutoTokenizer.from_pretrained(base_id)
    print(f"[paraphraser] Tokenizer geladen, vocab={len(tok)}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    print(f"[paraphraser] Base-Model geladen", flush=True)

    model = PeftModel.from_pretrained(base, adapter_id)
    model.eval()
    print(f"[paraphraser] StealthRL-Adapter aufgespielt", flush=True)

    PROMPT = ("Please paraphrase the following text while maintaining its meaning and style. "
              "Preserve every source claim, keep the paraphrase close to the original length, "
              "do not summarize, do not add new details, and output only the paraphrased text "
              "without any additional explanation.\n\n"
              "Original text:\n{text}\n\n"
              "Paraphrased text:")

    def paraphrase(text: str, max_new_tokens: int = 800, temperature: float = 0.8) -> str:
        prompt = PROMPT.format(text=text)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        decoded = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()

    return paraphrase


def main():
    out_fp = Path(__file__).resolve().parents[3] / "data" / "phase2" / "smoke" / "stealthrl_editlens.jsonl"
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    arts = load_eval_articles(limit=5)
    print(f"=== Smoke: {len(arts)} Casdorff-Held-out-Artikel ===", flush=True)

    try:
        editlens_score = setup_editlens_detector()
    except Exception as e:
        print(f"[detector] EditLens FAILED ({e!r}), fallback MAGE", flush=True)
        editlens_score = None

    try:
        mage_score = setup_mage_detector()
    except Exception as e:
        print(f"[detector] MAGE FAILED ({e!r})", flush=True)
        mage_score = None

    paraphrase = setup_stealthrl_paraphraser()

    rows = []
    for i, art in enumerate(arts):
        print(f"\n--- Artikel {i+1}/{len(arts)} ---", flush=True)
        print(f"    {art['datum']} — {art['titel']}", flush=True)
        t0 = time.time()
        orig = art["volltext"]

        # Pre-Scores
        pre_editlens = editlens_score(orig) if editlens_score else None
        pre_mage = mage_score(orig) if mage_score else None
        print(f"    pre-EditLens: {pre_editlens}, pre-MAGE: {pre_mage}", flush=True)

        # Paraphrase
        para = paraphrase(orig, max_new_tokens=min(800, len(orig) // 3 + 100))
        print(f"    paraphrased: {len(para)} chars (orig {len(orig)})", flush=True)

        # Post-Scores
        post_editlens = editlens_score(para) if editlens_score else None
        post_mage = mage_score(para) if mage_score else None
        print(f"    post-EditLens: {post_editlens}, post-MAGE: {post_mage}", flush=True)
        print(f"    duration: {time.time()-t0:.1f}s", flush=True)

        rows.append({
            "doc_id": art["doc_id"],
            "datum": art["datum"],
            "titel": art["titel"],
            "orig_chars": len(orig),
            "para_chars": len(para),
            "pre_editlens": pre_editlens,
            "post_editlens": post_editlens,
            "pre_mage": pre_mage,
            "post_mage": post_mage,
            "paraphrased_text": para,
            "duration_s": round(time.time() - t0, 1),
        })

    with out_fp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n=== Smoke fertig: {out_fp}")


if __name__ == "__main__":
    main()
