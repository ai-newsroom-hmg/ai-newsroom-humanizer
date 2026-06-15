"""Post-Score-Skript fuer Phase-2-Smoke-Output.

Liest die Paraphrasen aus stealthrl_editlens.jsonl + scort sie mit
Hello-SimpleAI/chatgpt-detector-roberta (publik verfuegbar, transformers
5.9-kompatibel, RoBERTa-base).

Labels: {0: 'Human', 1: 'ChatGPT'}. Wir reporten die ChatGPT-Probability als
"fraction_ai_proxy" — analog zu Pangram fraction_ai.

Auch gleich BGE-M3-Faithfulness-Check zwischen Original und Paraphrase.

Output: data/phase2/smoke/stealthrl_scored.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    in_fp = Path(__file__).resolve().parents[3] / "data" / "phase2" / "smoke" / "stealthrl_editlens.jsonl"
    out_fp = in_fp.parent / "stealthrl_scored.jsonl"
    if not in_fp.exists():
        sys.exit(f"in fehlt: {in_fp}")

    rows = [json.loads(l) for l in in_fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"=== Score {len(rows)} Paraphrasen ===", flush=True)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    import torch

    print("[detector] chatgpt-detector-roberta laden...", flush=True)
    DET = "Hello-SimpleAI/chatgpt-detector-roberta"
    det_tok = AutoTokenizer.from_pretrained(DET)
    det_m = AutoModelForSequenceClassification.from_pretrained(DET, torch_dtype=torch.float32)
    det_m.eval()
    # Label-Map verifizieren
    id2label = det_m.config.id2label
    print(f"[detector] Labels: {id2label}", flush=True)
    chatgpt_idx = next((i for i, l in id2label.items() if "chat" in str(l).lower() or "ai" in str(l).lower() or "machine" in str(l).lower()), 1)

    print("[bge-m3] Faithfulness-Embedding laden...", flush=True)
    bge = SentenceTransformer("BAAI/bge-m3")

    def detect(text: str) -> dict:
        with torch.no_grad():
            inputs = det_tok(text, return_tensors="pt", truncation=True, max_length=512)
            logits = det_m(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        if not isinstance(probs, list):
            probs = [probs]
        return {
            "fraction_ai_proxy": probs[chatgpt_idx] if len(probs) > chatgpt_idx else None,
            "fraction_human_proxy": probs[1 - chatgpt_idx] if len(probs) > 1 else None,
            "probs": probs,
        }

    out_rows = []
    for i, r in enumerate(rows):
        orig = (r.get("orig_text") or "")
        # Falls orig_text nicht da: aus DB nachladen waere noetig — wir lesen aus
        # paraphrased_text + nutzen die Frage „wie sicher ist Detector auf der Paraphrase?"
        para = r.get("paraphrased_text") or ""
        if not para:
            print(f"  [{i+1}] keine Paraphrase, skip", flush=True)
            continue

        post = detect(para)

        # Faithfulness: BGE-M3-Cosine (wir brauchen orig_text dafuer — naechste Iteration)
        post["chars"] = len(para)
        post["words"] = len(para.split())

        out_rows.append({
            "doc_id": r["doc_id"],
            "datum": r["datum"],
            "titel": r["titel"],
            "para_chars": r.get("para_chars"),
            "scored_post": post,
            "duration_paraphrase_s": r.get("duration_s"),
        })

        print(f"  [{i+1}/{len(rows)}] {r['titel'][:60]} | "
              f"fraction_ai_proxy={post['fraction_ai_proxy']:.3f}", flush=True)

    with out_fp.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== Output: {out_fp}")

    # Aggregat
    valid = [r for r in out_rows if r["scored_post"]["fraction_ai_proxy"] is not None]
    if valid:
        scores = [r["scored_post"]["fraction_ai_proxy"] for r in valid]
        n_under_02 = sum(1 for s in scores if s < 0.2)
        print(f"\n=== Bypass-Quote (fraction_ai_proxy < 0.2): {n_under_02}/{len(valid)} "
              f"= {n_under_02/len(valid):.0%}")
        print(f"=== Mean fraction_ai_proxy: {sum(scores)/len(scores):.3f}")
        print(f"=== Verteilung: {sorted(round(s, 3) for s in scores)}")


if __name__ == "__main__":
    main()
