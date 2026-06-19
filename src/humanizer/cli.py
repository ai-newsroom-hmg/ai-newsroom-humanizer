"""humanize — generisches CLI fuer den Humanizer (v0.2).

Modes:
  bestofn  (Default seit v0.2, 2026-06-18): Mistral-3.2 + Best-of-N + BGE-Filter
           + Pangram-Live-Rank. ADR-008 Phase-3a Methode (42 % Doc-Bypass auf
           Casdorff @ n=24). Empfohlen für deutsche Texte.
  loop     (Legacy, ADR-007 Phase 2): Sonnet-Iterative-Loop mit Proxy-Reward.
           4-21 % Bypass je nach Text. --legacy aktiviert diesen Pfad.

Beispiele:
  humanize artikel.txt                       # bestofn, 24 Varianten, Pangram-Rank
  humanize artikel.txt -o human.txt --variants 50
  humanize lange-meinung.txt --chunked       # auto bei >4000 chars sowieso
  cat artikel.txt | humanize - --variants 12 --no-rank-pangram   # billiger, kein API-Call
  humanize artikel.txt --legacy              # Rollback auf Sonnet-Loop
  humanize artikel.txt --json -o trace.json  # voller Trace mit allen Varianten

Rollback (UNBEDINGT-Skill 3 - always-one-version-rollback):
  git checkout v0.1-sonnet-loop -- src/humanizer/cli.py src/humanizer/core.py
  oder einfach: humanize <file> --legacy
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
    DEFAULT_THRESHOLD,
    HumanizeResult,
    ProxyScorer,
    humanize_text,
)
from humanizer.core_bestofn import (
    DEFAULT_BESTOFN_VARIANTS,
    DEFAULT_BGE_THRESHOLD,
    DEFAULT_CHUNK_THRESHOLD_CHARS,
    DEFAULT_PARAGRAPH_BGE_THRESHOLD,
    MODEL_MISTRAL_3_2,
    bge_similarity_batch,
    humanize_bestofn,
    humanize_chunked_bestofn,
)
from humanizer.env import detect_env


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


async def _pangram_pre(text: str, quiet: bool,
                       mock_cache: Optional[Path] = None) -> Optional[dict]:
    """Pre-Check des Originaltextes. mock_cache: in staging-mode → no API."""
    if mock_cache is None:
        if not os.environ.get("PANGRAM_API_KEY"):
            f = Path.home() / ".config" / "pangram" / "key"
            if f.exists():
                os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
        if not os.environ.get("PANGRAM_API_KEY"):
            _log("WARN: PANGRAM_API_KEY fehlt – Pre-Check übersprungen", quiet)
            return None
    from humanizer._pangram import PangramClient
    async with PangramClient(mock_cache_path=mock_cache) as pc:
        res = await pc.check_bulk([{"id": "pre", "text": text}])
    r = res.get("pre")
    if r and not r.error and r.fraction_ai is not None:
        return {"fraction_ai": r.fraction_ai, "prediction": r.prediction}
    return None


async def _run_bestofn(args: argparse.Namespace) -> int:
    quiet = args.quiet
    env = detect_env(args.env)
    orig = _read_input(args.input).strip()
    if not orig:
        sys.exit("Input ist leer")

    use_chunked = args.chunked or (len(orig) > DEFAULT_CHUNK_THRESHOLD_CHARS and not args.no_auto_chunk)
    _log(f"=== palimpsest [env={env.name}]: {len(orig)} chars, variants={args.variants}, "
         f"chunked={use_chunked}, rank_pangram={not args.no_rank_pangram} ===", quiet)
    if env.is_staging:
        _log(f"    STAGING: Pangram-Cache={env.pangram_cache_path}", quiet)
    mock_cache = env.pangram_cache_path if env.is_staging else None

    pre_info = None
    if args.eval or args.pre_check:
        _log("--- Pangram-Pre ---", quiet)
        pre_info = await _pangram_pre(orig, quiet, mock_cache=mock_cache)
        if pre_info:
            _log(f"    pre: fraction_ai={pre_info['fraction_ai']:.3f} "
                 f"prediction={pre_info['prediction']}", quiet)
            if pre_info["fraction_ai"] < 0.2:
                _log("    Text ist bereits Pangram-Human (<0.2) – Skip Humanizer.", quiet)
                _write_output(orig, args.out)
                return 0

    client = ORClient(model=args.model)
    t0 = time.time()

    def on_progress(msg: str) -> None:
        _log(f"  {msg}", quiet)

    if use_chunked:
        res = await humanize_chunked_bestofn(
            orig, client=client,
            n_variants_per_chunk=max(8, args.variants // 4),
            bge_threshold=DEFAULT_PARAGRAPH_BGE_THRESHOLD,
            rank_by_pangram=not args.no_rank_pangram,
            pangram_mock_cache=mock_cache,
            on_progress=on_progress,
        )
    else:
        res = await humanize_bestofn(
            orig, client=client,
            n_variants=args.variants,
            bge_threshold=args.bge_threshold,
            rank_by_pangram=not args.no_rank_pangram,
            pangram_pre=(pre_info or {}).get("fraction_ai"),
            pangram_mock_cache=mock_cache,
            on_progress=on_progress,
        )

    _log(f"--- Best-of-{res.n_generated} fertig: faithful={res.n_faithful} "
         f"bypass={res.n_bypass} BGE-Sim={res.bge_sim:.3f} "
         f"pangram_post={res.pangram_post} ({res.duration_s:.0f}s, "
         f"${res.total_cost_usd:.3f}, stop={res.stopped_reason}) ---", quiet)

    eval_block: dict = {}
    if args.eval:
        if res.pangram_post is None and res.final_text != orig:
            _log("--- Eval: Pangram-Post ---", quiet)
            post = await _pangram_pre(res.final_text, quiet, mock_cache=mock_cache)
            if post:
                res.pangram_post = post["fraction_ai"]
                _log(f"    post: fraction_ai={post['fraction_ai']:.3f} "
                     f"prediction={post['prediction']}", quiet)
        if res.chunked or res.bge_sim == 0.0:
            res.bge_sim = bge_similarity_batch(orig, [res.final_text])[0]
        eval_block = {
            "pangram_pre": pre_info,
            "pangram_post_fraction_ai": res.pangram_post,
            "bge_similarity": round(res.bge_sim, 4),
            "bypass": (res.pangram_post is not None and res.pangram_post < 0.2),
            "faithful": res.bge_sim >= 0.85,
        }
        _log(f"    eval: bypass={eval_block['bypass']} faithful={eval_block['faithful']}", quiet)

    if args.json:
        payload = {
            "mode": "bestofn",
            "model": args.model,
            "orig_text": res.orig_text,
            "final_text": res.final_text,
            "n_generated": res.n_generated,
            "n_faithful": res.n_faithful,
            "n_bypass": res.n_bypass,
            "bge_sim": res.bge_sim,
            "pangram_pre": (pre_info or {}).get("fraction_ai"),
            "pangram_post": res.pangram_post,
            "duration_s": res.duration_s,
            "total_cost_usd": res.total_cost_usd,
            "stopped_reason": res.stopped_reason,
            "chunked": res.chunked,
            "chunk_results": res.chunk_results,
            "variants": [
                {"idx": v.idx, "temp": v.temp, "bge_sim": round(v.bge_sim, 4),
                 "pangram_fraction_ai": v.pangram_fraction_ai,
                 "error": v.error, "chars": len(v.text)}
                for v in res.variants
            ],
            "eval": eval_block or None,
        }
        _write_output(json.dumps(payload, ensure_ascii=False, indent=2), args.out)
    else:
        _write_output(res.final_text, args.out)

    _log(f"=== done in {time.time() - t0:.0f}s ===", quiet)
    return 0


async def _run_legacy(args: argparse.Namespace) -> int:
    """Legacy-Pfad: Sonnet + Phase-2-Proxy-Loop (ADR 007). Rollback-Anker v0.1-sonnet-loop."""
    quiet = args.quiet
    orig = _read_input(args.input).strip()
    if not orig:
        sys.exit("Input ist leer")

    _log(f"=== humanize (legacy/loop): {len(orig)} chars, "
         f"threshold={args.threshold} max_iters={args.max_iters} "
         f"variants={args.variants} ===", quiet)
    _log("--- Proxy laden (BGE-M3 + MLP) ---", quiet)
    proxy = ProxyScorer()
    _log(f"    val_MAE={proxy.config.get('best_val_mae', '?'):.4f} device={proxy.device}", quiet)

    legacy_model = args.model if args.model != MODEL_MISTRAL_3_2 else MODEL_DEFAULT
    client = ORClient(model=legacy_model)
    t0 = time.time()

    def on_prog(it: int, score: float, n: int) -> None:
        _log(f"  iter {it}: proxy={score:.3f} (best of {n})", quiet)

    res: HumanizeResult = await humanize_text(
        orig, proxy=proxy, client=client,
        threshold=args.threshold, max_iters=args.max_iters,
        n_variants=args.variants, on_progress=on_prog,
    )

    _log(f"--- Loop fertig: proxy {res.proxy_score_pre:.3f} -> {res.proxy_score_post:.3f} "
         f"({res.iterations_run} iter, {res.duration_s:.0f}s, "
         f"${res.total_cost_usd:.3f}, stop={res.stopped_reason}) ---", quiet)

    if args.json:
        payload = {
            "mode": "legacy_loop",
            "model": legacy_model,
            "orig_text": res.orig_text,
            "final_text": res.final_text,
            "proxy_score_pre": res.proxy_score_pre,
            "proxy_score_post": res.proxy_score_post,
            "iterations_run": res.iterations_run,
            "duration_s": res.duration_s,
            "total_cost_usd": res.total_cost_usd,
            "stopped_reason": res.stopped_reason,
            "history": res.history,
        }
        _write_output(json.dumps(payload, ensure_ascii=False, indent=2), args.out)
    else:
        _write_output(res.final_text, args.out)

    _log(f"=== done in {time.time() - t0:.0f}s ===", quiet)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name if sys.argv else "palimpsest",
        description="Palimpsest — überschreibt KI-Output bis das Detektor-Signal verschwindet. "
                    "v0.2 Mistral-3.2 + Best-of-N + BGE-Multi-Chunk + Pangram-Rank (ADR 008/009).",
    )
    ap.add_argument("input", help="Pfad zur Textdatei oder '-' fuer stdin")
    ap.add_argument("-o", "--out", default=None, help="Output-Datei (Default: stdout). '-' = stdout.")
    ap.add_argument("--env", choices=("dev", "staging"), default=None,
                    help="dev (default): live OpenRouter+Pangram. staging: Pangram-Cache-Only (no API-Cost). "
                         "Overrides PALIMPSEST_ENV.")
    ap.add_argument("--mode", choices=("bestofn", "loop"), default="bestofn",
                    help="bestofn (Default v0.2) oder loop (Legacy Phase-2)")
    ap.add_argument("--legacy", action="store_true",
                    help="Alias: --mode loop + Sonnet-Model (Rollback-Pfad, UNBEDINGT-Skill 3)")
    ap.add_argument("--model", default=None,
                    help=f"OpenRouter-Modell. Default bestofn: {MODEL_MISTRAL_3_2}, loop: {MODEL_DEFAULT}")
    ap.add_argument("--variants", type=int, default=DEFAULT_BESTOFN_VARIANTS,
                    help=f"Anzahl Best-of-N Varianten (Default {DEFAULT_BESTOFN_VARIANTS})")
    ap.add_argument("--bge-threshold", type=float, default=DEFAULT_BGE_THRESHOLD,
                    help=f"BGE-Sim Faithfulness-Schwelle (Default {DEFAULT_BGE_THRESHOLD})")
    ap.add_argument("--chunked", action="store_true",
                    help="Paragraph-wise Best-of-N (auto bei Texten >4000 chars)")
    ap.add_argument("--no-auto-chunk", action="store_true",
                    help="Auto-chunked bei >4000 chars deaktivieren")
    ap.add_argument("--no-rank-pangram", action="store_true",
                    help="Kein Pangram-Live-Ranking (spart ~$1-2 pro Run; ranking per BGE-Sim)")
    ap.add_argument("--pre-check", action="store_true",
                    help="Pangram-Pre-Check vor Humanize (skip wenn schon human)")
    ap.add_argument("--eval", action="store_true",
                    help="Pre/Post-Pangram + BGE-Faithfulness am Ende (~$0.10)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"[loop] Proxy-Score Stop-Schwelle (Default {DEFAULT_THRESHOLD})")
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS,
                    help=f"[loop] Max Iterationen (Default {DEFAULT_MAX_ITERS})")
    ap.add_argument("--json", action="store_true",
                    help="Vollen JSON-Trace ausgeben statt nur des Textes")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Keine Progress-Logs auf stderr")
    args = ap.parse_args()

    if args.legacy:
        args.mode = "loop"
    if args.model is None:
        args.model = MODEL_DEFAULT if args.mode == "loop" else MODEL_MISTRAL_3_2

    if args.mode == "loop":
        sys.exit(asyncio.run(_run_legacy(args)))
    else:
        sys.exit(asyncio.run(_run_bestofn(args)))


if __name__ == "__main__":
    main()
