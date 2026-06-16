"""Generischer Humanizer-Kern (Phase-2-Loop, wiederverwendbar).

Best-of-N Sonnet-Varianten + Proxy-Reward, iterativ bis Proxy-Score < THRESHOLD
oder MAX_ITERS erreicht. Funktioniert auf beliebigem Text — nicht an Casdorff
gebunden. Verwendet von p3_detector_aware_loop.py (Batch-Eval) und
humanizer.cli (Einzeltext).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from humanizer._openrouter import MODEL_DEFAULT, ORClient

ROOT = Path(__file__).resolve().parents[2]
PROXY_MODEL = ROOT / "data" / "phase2" / "proxy_model.pt"
PROXY_CONFIG = ROOT / "data" / "phase2" / "proxy_config.json"

DEFAULT_THRESHOLD = 0.20
DEFAULT_MAX_ITERS = 5
DEFAULT_N_VARIANTS = 10
DEFAULT_TEMPS = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1)

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
    """BGE-M3 + 2-Layer-MLP-Head (val_MAE 0.29, Phase 2)."""

    def __init__(self, model_path: Path = PROXY_MODEL, config_path: Path = PROXY_CONFIG):
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Proxy-Artefakte fehlen: {model_path} / {config_path}. "
                f"Sync von ruediger: "
                f"rsync ruediger:Projects/ai-newsroom-humanizer/data/phase2/proxy_* "
                f"data/phase2/"
            )
        import torch
        from sentence_transformers import SentenceTransformer

        cfg = json.loads(config_path.read_text())
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.enc = SentenceTransformer(cfg["embed_model"], device=self.device)

        sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))
        from p2_train_proxy import ProxyHead  # noqa: E402

        self.model = ProxyHead(dim_in=cfg["embed_dim"], hidden=cfg["hidden"]).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.torch = torch
        self.config = cfg

    def score(self, texts: list[str]) -> list[float]:
        with self.torch.no_grad():
            emb = self.enc.encode(
                texts, convert_to_tensor=True, normalize_embeddings=True,
                batch_size=8, show_progress_bar=False,
            ).to(self.device)
            return self.model(emb).cpu().tolist()


@dataclass
class HumanizeResult:
    orig_text: str
    final_text: str
    proxy_score_pre: float
    proxy_score_post: float
    iterations_run: int
    history: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_s: float = 0.0
    stopped_reason: str = ""


async def _gen_variant(client: ORClient, system: str, user: str, temp: float) -> dict:
    out = await client.complete(system, user, temperature=temp, max_tokens=3000)
    return {"text": out["text"].strip(), "cost": out["cost_usd"]}


async def humanize_text(
    text: str,
    *,
    proxy: ProxyScorer,
    client: Optional[ORClient] = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_iters: int = DEFAULT_MAX_ITERS,
    n_variants: int = DEFAULT_N_VARIANTS,
    temps: tuple[float, ...] = DEFAULT_TEMPS,
    on_progress=None,
) -> HumanizeResult:
    """Humanize a single text via Sonnet best-of-N + Proxy-Reward.

    on_progress: optional callback(iter_idx, best_score, n_variants_tested).
    """
    if client is None:
        client = ORClient(model=MODEL_DEFAULT)

    orig = text.strip()
    history = [{
        "iter": 0,
        "proxy_score": float(proxy.score([orig])[0]),
        "text_chars": len(orig),
        "source": "baseline",
    }]
    current_text: Optional[str] = None
    current_score = history[0]["proxy_score"]
    total_cost = 0.0
    t0 = time.time()
    stopped = "max_iters"

    if current_score < threshold:
        stopped = "already_below_threshold"
        return HumanizeResult(
            orig_text=orig, final_text=orig,
            proxy_score_pre=current_score, proxy_score_post=current_score,
            iterations_run=0, history=history,
            total_cost_usd=0.0, duration_s=round(time.time() - t0, 1),
            stopped_reason=stopped,
        )

    for it in range(1, max_iters + 1):
        if it == 1:
            user = USER_FIRST.format(text=orig)
        else:
            user = USER_ITER.format(orig=orig, score=current_score, current=current_text)

        try:
            results = await asyncio.gather(*[
                _gen_variant(client, SYSTEM_PROMPT, user, temps[i % len(temps)])
                for i in range(n_variants)
            ])
        except Exception as e:
            history.append({"iter": it, "error": str(e)[:200]})
            stopped = f"error: {type(e).__name__}"
            break

        for r in results:
            total_cost += r["cost"]
        variants = [r["text"] for r in results]
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

        current_text = best_text
        current_score = best_score

        if on_progress:
            on_progress(it, current_score, n_variants)

        if current_score < threshold:
            stopped = "threshold_reached"
            break

    return HumanizeResult(
        orig_text=orig,
        final_text=current_text or orig,
        proxy_score_pre=history[0]["proxy_score"],
        proxy_score_post=current_score,
        iterations_run=len(history) - 1,
        history=history,
        total_cost_usd=round(total_cost, 4),
        duration_s=round(time.time() - t0, 1),
        stopped_reason=stopped,
    )
