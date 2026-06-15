"""Phase 2 - Schritt 3: Detector-aware Sonnet-Loop mit Proxy-Reward.

Generisches Bypass-Tool. Per Default arbeitet es auf data/phase2-training-pool/eval.jsonl,
kann aber auch fuer beliebige Input-Texte aufgerufen werden via CLI.

Pro Artikel:
  Iteration k:
    1. Sonnet generiert N=10 Varianten parallel (verschiedene temperatures)
    2. Proxy bewertet alle 10 -> beste hat niedrigsten Proxy-Score
    3. Wenn Proxy-Score < THRESHOLD: stop
    4. Sonst: beste als "current" fuer naechste Iter
  Max ITERS=5.

Stop-Kriterien:
  - proxy_score < 0.2
  - max_iter erreicht

Output: data/phase2/loop_results.jsonl
Kosten: 24 × 10 × 5 = max 1200 Sonnet-Calls (~$30), 0 Pangram-Calls (Proxy lokal)
Laufzeit: 3-6h
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _openrouter import ORClient, MODEL_DEFAULT  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
INP_DEFAULT = ROOT / "data" / "phase2-training-pool" / "eval.jsonl"
OUT = ROOT / "data" / "phase2" / "loop_results.jsonl"
PROXY_MODEL = ROOT / "data" / "phase2" / "proxy_model.pt"
PROXY_CONFIG = ROOT / "data" / "phase2" / "proxy_config.json"

THRESHOLD = 0.20
MAX_ITERS = 5
N_VARIANTS = 10
PARALLEL = 2
MODEL = MODEL_DEFAULT


SYSTEM_PROMPT = """Du bist ein erfahrener deutscher Journalist. Schreibe den
folgenden Text so um, dass er menschlich klingt — variable Satzlaengen,
gelegentliche Brueche im Rhythmus, idiomatische Wendungen, eigene Wortwahl.
Inhalt, Fakten, Namen, Zahlen, Zitate bleiben EXAKT erhalten.

Vermeide:
- Floskel-Anschluesse ('daher', 'in diesem Kontext', 'vor diesem Hintergrund')
- Erstens / Zweitens / Drittens-Strukturen
- Drei parallele Adjektive ('klar, transparent, nachvollziehbar')
- Glatte Aufzaehlungen

Antworte NUR mit dem umgeschriebenen Text. Keine Vorrede."""

USER_FIRST = "Originaltext:\n\n{text}\n\nMenschlich umgeschrieben:"
USER_ITER = """Originaltext:

{orig}

Aktuelle Fassung (Proxy-Score = {score:.3f} -- noch zu maschinell):

{current}

Schreibe radikaler menschlich um. Inhalt 100% treu zum Original. NUR Stil."""


class ProxyScorer:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        cfg = json.loads(PROXY_CONFIG.read_text())
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.enc = SentenceTransformer(cfg["embed_model"], device=self.device)

        # Proxy-Head reload
        from p2_train_proxy import ProxyHead  # noqa: E402
        sys.path.insert(0, str(ROOT / "src" / "humanizer" / "training"))
        from p2_train_proxy import ProxyHead
        self.model = ProxyHead(dim_in=cfg["embed_dim"], hidden=cfg["hidden"]).to(self.device)
        state = torch.load(PROXY_MODEL, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.torch = torch

    def score(self, texts: list[str]) -> list[float]:
        with self.torch.no_grad():
            emb = self.enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                                  batch_size=8, show_progress_bar=False).to(self.device)
            return self.model(emb).cpu().tolist()


async def gen_variant(client: ORClient, system: str, user: str, temp: float) -> dict:
    out = await client.complete(system, user, temperature=temp, max_tokens=3000)
    return {"text": out["text"].strip(), "cost": out["cost_usd"]}


async def loop_one(art: dict, client: ORClient, proxy: ProxyScorer, idx: int) -> dict:
    orig = art["volltext"]
    history = [{
        "iter": 0,
        "proxy_score": float(proxy.score([orig])[0]),
        "text_chars": len(orig),
        "source": "baseline",
    }]

    current = None
    current_score = history[0]["proxy_score"]
    total_cost = 0.0
    t0 = time.time()

    # Temperaturen-Sweep fuer Diversitaet
    temps = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]

    for it in range(1, MAX_ITERS + 1):
        if current_score < THRESHOLD:
            break

        if it == 1:
            user = USER_FIRST.format(text=orig)
        else:
            user = USER_ITER.format(orig=orig, score=current_score, current=current)

        # N Varianten parallel
        try:
            results = await asyncio.gather(*[
                gen_variant(client, SYSTEM_PROMPT, user, temps[i % len(temps)])
                for i in range(N_VARIANTS)
            ])
        except Exception as e:
            history.append({"iter": it, "error": str(e)[:200]})
            break

        for r in results:
            total_cost += r["cost"]
        variants = [r["text"] for r in results]

        # Proxy bewertet alle
        scores = proxy.score(variants)
        idx_best = min(range(len(scores)), key=lambda i: scores[i])
        best_text = variants[idx_best]
        best_score = scores[idx_best]

        history.append({
            "iter": it,
            "proxy_score": float(best_score),
            "variants_tested": len(scores),
            "min_score": float(min(scores)),
            "max_score": float(max(scores)),
            "text_chars": len(best_text),
            "variant_cost": round(sum(r["cost"] for r in results), 6),
        })

        current = best_text
        current_score = best_score

    return {
        "doc_id": art.get("doc_id"),
        "datum": art.get("datum"),
        "titel": art.get("titel"),
        "quelle": art.get("quelle"),
        "orig_text": orig,
        "final_text": current or orig,
        "proxy_score_pre": history[0]["proxy_score"],
        "proxy_score_post": current_score,
        "iterations_run": len(history) - 1,
        "history": history,
        "total_cost_usd": round(total_cost, 4),
        "duration_s": round(time.time() - t0, 1),
    }


async def main():
    inp = Path(os.environ.get("HUMANIZER_INPUT", str(INP_DEFAULT)))
    if not inp.exists():
        sys.exit(f"INP fehlt: {inp}")
    arts = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Erwarte 'volltext' Feld (DB-Schema) — falls 'text': renaming
    for a in arts:
        if "volltext" not in a:
            a["volltext"] = a.get("text") or a.get("body") or a.get("content") or ""

    print(f"=== Detector-aware Loop: {len(arts)} Artikel ===", flush=True)
    print(f"    THRESHOLD: {THRESHOLD}, MAX_ITERS: {MAX_ITERS}, N_VARIANTS: {N_VARIANTS}", flush=True)

    print(f"\n--- Proxy laden ---", flush=True)
    proxy = ProxyScorer()
    cfg = json.loads(PROXY_CONFIG.read_text())
    print(f"    Best val_MAE: {cfg.get('best_val_mae', '?'):.4f}", flush=True)

    client = ORClient(model=MODEL)
    sem = asyncio.Semaphore(PARALLEL)
    results = [None] * len(arts)
    done = 0
    total_cost = 0.0
    t_start = time.time()

    async def worker(i, art):
        nonlocal done, total_cost
        async with sem:
            r = await loop_one(art, client, proxy, i)
            results[i] = r
            done += 1
            total_cost += r.get("total_cost_usd", 0)
            elapsed = time.time() - t_start
            pre = r["proxy_score_pre"]
            post = r["proxy_score_post"]
            tag = "OK" if post < THRESHOLD else "FAIL"
            print(f"  [{done}/{len(arts)}] {tag:>4} {art.get('datum','?')} "
                  f"proxy {pre:.3f} -> {post:.3f} "
                  f"({r['iterations_run']} iter, {r['duration_s']:.0f}s, "
                  f"${r['total_cost_usd']:.3f}) "
                  f"[total ${total_cost:.2f}, elapsed {elapsed:.0f}s]", flush=True)

    await asyncio.gather(*(worker(i, a) for i, a in enumerate(arts)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_proxy_success = sum(1 for r in results if r["proxy_score_post"] < THRESHOLD)
    print(f"\n=== FERTIG ===")
    print(f"  Proxy-Bypass (proxy < {THRESHOLD}): {n_proxy_success}/{len(results)}")
    print(f"  Total Sonnet-Kosten: ${total_cost:.4f}")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
