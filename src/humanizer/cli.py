"""humanize — generisches CLI fuer den Phase-2-Humanizer-Loop.

Beispiele:
  humanize artikel.txt                   # auf stdout
  humanize artikel.txt -o human.txt      # in Datei
  cat artikel.txt | humanize -           # stdin
  humanize artikel.txt --eval            # mit Pre/Post-Pangram + BGE-Sim
  humanize artikel.txt --max-iters 3 --variants 6 --threshold 0.25 --json

Reward: Phase-2-Proxy (BGE-M3 + MLP), lokal auf MPS (Apple Silicon).
Default-Modell: anthropic/claude-sonnet-4.5 via OpenRouter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from humanizer._openrouter import MODEL_DEFAULT, ORClient
from humanizer.core import (
    DEFAULT_MAX_ITERS,
    DEFAULT_N_VARIANTS,
    DEFAULT_THRESHOLD,
    HumanizeResult,
    ProxyScorer,
    humanize_text,
)


def _read_input(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    p = Path(arg).expanduser()
    if not p.exists():
        sys.exit(f"Input nicht gefunden: {p}")
    return p.read_text(encoding="utf-8")


def _write_output(text: str, out: Optional[str]) -> None:
    if out is None or out == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    p = Path(out).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


async def _pangram_score(text_pairs: list[tuple[str, str]], quiet: bool) -> dict[str, dict]:
    if not os.environ.get("PANGRAM_API_KEY"):
        key_file = Path.home() / ".config" / "pangram" / "key"
        if key_file.exists():
            os.environ["PANGRAM_API_KEY"] = key_file.read_text().strip()
    if not os.environ.get("PANGRAM_API_KEY"):
        _log("WARN: PANGRAM_API_KEY fehlt — Eval-Skip", quiet)
        return {}

    from humanizer._pangram import PangramClient

    async with PangramClient() as pc:
        items = [{"id": tid, "text": txt} for tid, txt in text_pairs]
        res = await pc.check_bulk(items)
    out: dict[str, dict] = {}
    for tid, _ in text_pairs:
        r = res.get(tid)
        if r and not r.error and r.fraction_ai is not None:
            out[tid] = {
                "fraction_ai": r.fraction_ai,
                "fraction_human": r.fraction_human,
                "prediction": r.prediction,
            }
        else:
            out[tid] = {"error": (r.error if r else "no-result")}
    return out


def _bge_similarity(a: str, b: str) -> float:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=device)
    with torch.no_grad():
        embs = enc.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return float((embs[0] * embs[1]).sum().item())


async def _run(args: argparse.Namespace) -> int:
    quiet = args.quiet
    orig_text = _read_input(args.input).strip()
    if not orig_text:
        sys.exit("Input ist leer")

    _log(f"=== humanize: {len(orig_text)} chars ===", quiet)
    _log(f"    threshold={args.threshold} max_iters={args.max_iters} variants={args.variants}", quiet)

    _log("--- Proxy laden (BGE-M3 + MLP) ---", quiet)
    proxy = ProxyScorer()
    _log(f"    val_MAE={proxy.config.get('best_val_mae', '?'):.4f} device={proxy.device}", quiet)

    client = ORClient(model=args.model)
    t_total = time.time()

    def on_progress(it: int, score: float, n: int) -> None:
        _log(f"  iter {it}: proxy={score:.3f} (best of {n})", quiet)

    res: HumanizeResult = await humanize_text(
        orig_text,
        proxy=proxy,
        client=client,
        threshold=args.threshold,
        max_iters=args.max_iters,
        n_variants=args.variants,
        on_progress=on_progress,
    )

    _log(
        f"--- Loop fertig: proxy {res.proxy_score_pre:.3f} -> {res.proxy_score_post:.3f} "
        f"({res.iterations_run} iter, {res.duration_s:.0f}s, ${res.total_cost_usd:.3f}, "
        f"stop={res.stopped_reason}) ---",
        quiet,
    )

    eval_block: dict = {}
    if args.eval:
        _log("--- Eval: Pangram pre/post + BGE-Sim ---", quiet)
        pangram = await _pangram_score(
            [("pre", res.orig_text), ("post", res.final_text)], quiet
        )
        sim = _bge_similarity(res.orig_text, res.final_text)
        eval_block = {
            "pangram_pre": pangram.get("pre", {}),
            "pangram_post": pangram.get("post", {}),
            "bge_similarity": round(sim, 4),
            "bypass": (pangram.get("post", {}).get("fraction_ai") or 1.0) < 0.2,
            "faithful": sim >= 0.85,
        }
        _log(
            f"    pangram: {eval_block['pangram_pre'].get('fraction_ai')} -> "
            f"{eval_block['pangram_post'].get('fraction_ai')} | "
            f"BGE-Sim={eval_block['bge_similarity']} | "
            f"bypass={eval_block['bypass']} faithful={eval_block['faithful']}",
            quiet,
        )

    if args.json:
        payload = {
            "orig_text": res.orig_text,
            "final_text": res.final_text,
            "proxy_score_pre": res.proxy_score_pre,
            "proxy_score_post": res.proxy_score_post,
            "iterations_run": res.iterations_run,
            "duration_s": res.duration_s,
            "total_cost_usd": res.total_cost_usd,
            "stopped_reason": res.stopped_reason,
            "history": res.history,
            "eval": eval_block or None,
        }
        out_str = json.dumps(payload, ensure_ascii=False, indent=2)
        _write_output(out_str, args.out)
    else:
        _write_output(res.final_text, args.out)

    _log(f"=== done in {time.time() - t_total:.0f}s ===", quiet)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="humanize",
        description="AI-Text in menschlich-klingenden Text umschreiben "
                    "(Phase-2-Loop, Sonnet + Proxy-Reward).",
    )
    ap.add_argument("input", help="Pfad zur Textdatei oder '-' fuer stdin")
    ap.add_argument("-o", "--out", default=None,
                    help="Output-Datei (Default: stdout). '-' = stdout.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Proxy-Score Stop-Schwelle (Default {DEFAULT_THRESHOLD})")
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS,
                    help=f"Maximale Iterationen (Default {DEFAULT_MAX_ITERS})")
    ap.add_argument("--variants", type=int, default=DEFAULT_N_VARIANTS,
                    help=f"Best-of-N Varianten pro Iteration (Default {DEFAULT_N_VARIANTS})")
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help=f"OpenRouter-Modell (Default {MODEL_DEFAULT})")
    ap.add_argument("--eval", action="store_true",
                    help="Pre/Post-Pangram-API + BGE-Faithfulness am Ende (~$0.10)")
    ap.add_argument("--json", action="store_true",
                    help="Vollen JSON-Trace ausgeben statt nur des Textes")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Keine Progress-Logs auf stderr")
    args = ap.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
