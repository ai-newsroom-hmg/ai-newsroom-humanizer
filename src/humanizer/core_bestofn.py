"""Best-of-N Humanizer-Pipeline (Mistral-3.2 + BGE-Filter + Pangram-Rank).

ADR-008 Phase-3a Methode (42 % Doc-Bypass auf Casdorff @ n=24):
- Default-Model: Mistral-Small-3.2-24b-instruct via OpenRouter
- Parallel Best-of-N mit Temperatur-Sweep (0.85-1.15)
- BGE-M3-Sim ≥ threshold als Faithfulness-Gate (default 0.85)
- Pangram-Live-Score als Ranking-Signal (lowest fraction_ai wins)
- Chunked-Mode für Texte > 4000 chars (paragraph-wise, BGE pro Absatz ≥ 0.80)

NEU in v0.2 (2026-06-18). Legacy-Pfad in core.humanize_text bleibt erhalten
(--legacy Flag).
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from humanizer._openrouter import ORClient

MODEL_MISTRAL_3_2 = "mistralai/mistral-small-3.2-24b-instruct"
MODEL_MISTRAL_3_2_BASE = "mistralai/mistral-small-3.2-24b"  # Pfad-D Beifang

DEFAULT_BESTOFN_VARIANTS = 24
DEFAULT_BESTOFN_TEMPS = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
DEFAULT_BGE_THRESHOLD = 0.85
DEFAULT_CHUNK_THRESHOLD_CHARS = 4000
DEFAULT_PARAGRAPH_BGE_THRESHOLD = 0.80
PANGRAM_BYPASS_THRESHOLD = 0.20

SYSTEM_PROMPT_BESTOFN = """Du bist ein erfahrener deutscher Journalist. Schreibe den
folgenden Text so um, dass er menschlich klingt — variable Satzlaengen,
gelegentliche Brueche im Rhythmus, idiomatische Wendungen, eigene Wortwahl.
Inhalt, Fakten, Namen, Zahlen, Zitate bleiben EXAKT erhalten.

Vermeide:
- Floskel-Anschluesse ('daher', 'in diesem Kontext', 'vor diesem Hintergrund')
- Erstens / Zweitens / Drittens-Strukturen
- Drei parallele Adjektive ('klar, transparent, nachvollziehbar')
- Glatte Aufzaehlungen

Antworte NUR mit dem umgeschriebenen Text. Keine Vorrede."""

USER_PROMPT = "Originaltext:\n\n{text}\n\nMenschlich umgeschrieben:"


@dataclass
class BestofNVariant:
    idx: int
    temp: float
    text: str
    bge_sim: float = 0.0
    pangram_fraction_ai: Optional[float] = None
    pangram_prediction: str = ""
    cost_usd: float = 0.0
    error: str = ""

    def is_faithful(self, threshold: float = DEFAULT_BGE_THRESHOLD) -> bool:
        return not self.error and self.bge_sim >= threshold

    def is_bypass(self) -> bool:
        return (self.pangram_fraction_ai is not None
                and self.pangram_fraction_ai < PANGRAM_BYPASS_THRESHOLD)


@dataclass
class HumanizeBestofNResult:
    orig_text: str
    final_text: str
    pangram_pre: Optional[float] = None
    pangram_post: Optional[float] = None
    bge_sim: float = 0.0
    variants: list[BestofNVariant] = field(default_factory=list)
    n_generated: int = 0
    n_faithful: int = 0
    n_bypass: int = 0
    total_cost_usd: float = 0.0
    duration_s: float = 0.0
    stopped_reason: str = ""
    chunked: bool = False
    chunk_results: list[dict] = field(default_factory=list)


_BGE_ENCODER = None
# Hard truncate für BGE-Encoding bei langen Texten — sonst MPS OOM
# (BGE-M3 max_seq=8192 tokens; bei 6+ Varianten × 8k Chars sprengt das MPS)
BGE_MAX_CHARS_PER_TEXT = 4000


def _get_bge_encoder():
    global _BGE_ENCODER
    if _BGE_ENCODER is None:
        import os
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        import torch
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _BGE_ENCODER = SentenceTransformer("BAAI/bge-m3", device=device)
    return _BGE_ENCODER


def _truncate(t: str, max_chars: int = BGE_MAX_CHARS_PER_TEXT) -> str:
    return t if len(t) <= max_chars else t[:max_chars]


def _chunk_text(t: str, chunk_size: int = BGE_MAX_CHARS_PER_TEXT) -> list[str]:
    return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)] or [t]


def _bge_encode_one(text: str):
    """Encode single text → normalized embedding tensor (MPS-safe, batch_size=1)."""
    import torch
    enc = _get_bge_encoder()
    with torch.no_grad():
        return enc.encode([text], convert_to_tensor=True, normalize_embeddings=True,
                          batch_size=1, show_progress_bar=False)[0]


def bge_similarity_batch(orig: str, candidates: list[str]) -> list[float]:
    """Cosine similarity orig vs each candidate, BGE-M3.

    Lange Texte: per-chunk-Sim (jedes orig-chunk vs best-match cand-chunk),
    Aggregation MIN (konservativ — der schwächste Absatz dominiert).
    Damit erwischen wir Token-Salat-Drift in späten Absätzen (siehe Smoke-Befund
    2026-06-19: BGE truncated → 0.94 trotz halluziniertem End-Drittel).

    Kurze Texte (<= BGE_MAX_CHARS): klassischer single-shot Vergleich.
    """
    if not candidates:
        return []
    sims: list[float] = []
    short_orig = len(orig) <= BGE_MAX_CHARS_PER_TEXT
    o_chunks = [_truncate(orig)] if short_orig else _chunk_text(orig)
    o_embs = [_bge_encode_one(c) for c in o_chunks]

    for cand in candidates:
        c_chunks = [_truncate(cand)] if len(cand) <= BGE_MAX_CHARS_PER_TEXT else _chunk_text(cand)
        c_embs = [_bge_encode_one(c) for c in c_chunks]
        # For each orig chunk: max-sim against any cand chunk; aggregate via min
        chunk_sims = []
        for oe in o_embs:
            best = max(float((oe * ce).sum().item()) for ce in c_embs)
            chunk_sims.append(best)
        sims.append(min(chunk_sims) if chunk_sims else 0.0)
    return sims


async def _gen_variant(client: ORClient, text: str, temp: float, idx: int,
                       max_tokens: int) -> BestofNVariant:
    try:
        out = await client.complete(
            SYSTEM_PROMPT_BESTOFN, USER_PROMPT.format(text=text),
            temperature=temp, max_tokens=max_tokens,
        )
        return BestofNVariant(idx=idx, temp=temp, text=out["text"].strip(),
                              cost_usd=out["cost_usd"])
    except Exception as e:
        return BestofNVariant(idx=idx, temp=temp, text="", error=str(e)[:200])


async def _pangram_check_variants(variants: list[BestofNVariant],
                                  bge_threshold: float,
                                  pangram_mock_cache: Optional[Path] = None) -> None:
    """In-place: setzt pangram_fraction_ai für jede faithful Variante.

    pangram_mock_cache: in staging-mode → load JSON cache, no API-calls.
    """
    if pangram_mock_cache is None:
        if not os.environ.get("PANGRAM_API_KEY"):
            f = Path.home() / ".config" / "pangram" / "key"
            if f.exists():
                os.environ["PANGRAM_API_KEY"] = f.read_text().strip()
        if not os.environ.get("PANGRAM_API_KEY"):
            return

    from humanizer._pangram import PangramClient
    items = [{"id": f"v{v.idx}", "text": v.text}
             for v in variants if v.is_faithful(bge_threshold)]
    if not items:
        return

    async with PangramClient(mock_cache_path=pangram_mock_cache) as pc:
        res = await pc.check_bulk(items)
    idx_to_v = {f"v{v.idx}": v for v in variants}
    for k, r in res.items():
        v = idx_to_v.get(k)
        if v and r and not r.error and r.fraction_ai is not None:
            v.pangram_fraction_ai = r.fraction_ai
            v.pangram_prediction = r.prediction


async def humanize_bestofn(
    text: str,
    *,
    client: Optional[ORClient] = None,
    n_variants: int = DEFAULT_BESTOFN_VARIANTS,
    temps: tuple[float, ...] = DEFAULT_BESTOFN_TEMPS,
    bge_threshold: float = DEFAULT_BGE_THRESHOLD,
    rank_by_pangram: bool = True,
    pangram_pre: Optional[float] = None,
    pangram_mock_cache: Optional[Path] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> HumanizeBestofNResult:
    """Whole-text Best-of-N: gen → BGE-Filter → Pangram-Rank → best variant.

    rank_by_pangram=False: ohne Pangram (lokaler Mode, kein API-Call); ranking nur per BGE-Sim.
    """
    if client is None:
        client = ORClient(model=MODEL_MISTRAL_3_2)

    orig = text.strip()
    t0 = time.time()
    max_tokens = max(3000, int(len(orig) * 0.6))

    if on_progress:
        on_progress(f"gen {n_variants} variants (max_tokens={max_tokens}, model={client.model})")

    tasks = [_gen_variant(client, orig, temps[i % len(temps)], i, max_tokens)
             for i in range(n_variants)]
    variants = await asyncio.gather(*tasks)
    n_ok = sum(1 for v in variants if not v.error and v.text)

    if on_progress:
        on_progress(f"  gen ok: {n_ok}/{n_variants}; BGE-Sim …")

    ok_vs = [v for v in variants if not v.error and v.text]
    if ok_vs:
        sims = bge_similarity_batch(orig, [v.text for v in ok_vs])
        for v, s in zip(ok_vs, sims, strict=True):
            v.bge_sim = s

    n_faithful = sum(1 for v in variants if v.is_faithful(bge_threshold))
    if on_progress:
        on_progress(f"  faithful (BGE>={bge_threshold}): {n_faithful}/{n_ok}")

    if rank_by_pangram and n_faithful > 0:
        if on_progress:
            mode = "MOCK" if pangram_mock_cache else "API"
            on_progress(f"  Pangram-Eval ({mode}) on {n_faithful} faithful …")
        await _pangram_check_variants(variants, bge_threshold, pangram_mock_cache)

    faithful_variants = [v for v in variants if v.is_faithful(bge_threshold)]
    if not faithful_variants:
        sorted_vs = sorted(ok_vs, key=lambda v: -v.bge_sim)
        best = sorted_vs[0] if sorted_vs else BestofNVariant(
            idx=-1, temp=0.0, text=orig, error="all variants failed")
        stopped = "no_faithful_fallback_to_best_bge"
    elif rank_by_pangram:
        scored = sorted(
            faithful_variants,
            key=lambda v: (v.pangram_fraction_ai if v.pangram_fraction_ai is not None else 1.0,
                           -v.bge_sim),
        )
        best = scored[0]
        stopped = "ranked_by_pangram"
    else:
        best = sorted(faithful_variants, key=lambda v: -v.bge_sim)[0]
        stopped = "ranked_by_bge"

    n_bypass = sum(1 for v in variants if v.is_bypass())

    return HumanizeBestofNResult(
        orig_text=orig,
        final_text=best.text or orig,
        pangram_pre=pangram_pre,
        pangram_post=best.pangram_fraction_ai,
        bge_sim=best.bge_sim,
        variants=variants,
        n_generated=n_variants,
        n_faithful=n_faithful,
        n_bypass=n_bypass,
        total_cost_usd=round(sum(v.cost_usd for v in variants), 4),
        duration_s=round(time.time() - t0, 1),
        stopped_reason=stopped,
    )


def split_paragraphs(text: str, min_chars: int = 400, max_chars: int = 2000) -> list[str]:
    """Robust paragraph-splitter: bevorzugt \\n\\n, fällt zurück auf \\n.

    Targets: chunks zwischen min_chars und max_chars. Texte aus iCloud-Sync
    haben oft \\n statt \\n\\n (Smoke-Befund 2026-06-19).
    """
    # Erstwahl: double-newline
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Falls die Aufteilung zu grob ist (chunks viel größer als max_chars),
    # auch auf single-newline aufteilen
    if any(len(p) > max_chars for p in raw):
        raw = [p.strip() for p in text.replace("\n\n", "\n").split("\n") if p.strip()]
    out: list[str] = []
    buf = ""
    for p in raw:
        if buf and len(buf) + len(p) > max_chars:
            out.append(buf)
            buf = p
        else:
            buf = (buf + "\n\n" + p) if buf else p
        if len(buf) >= min_chars and len(buf) >= max_chars * 0.8:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf) < min_chars:
            out[-1] = out[-1] + "\n\n" + buf
        else:
            out.append(buf)
    return out


async def humanize_chunked_bestofn(
    text: str,
    *,
    client: Optional[ORClient] = None,
    n_variants_per_chunk: int = 12,
    bge_threshold: float = DEFAULT_PARAGRAPH_BGE_THRESHOLD,
    rank_by_pangram: bool = True,
    pangram_mock_cache: Optional[Path] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> HumanizeBestofNResult:
    """Paragraph-wise Best-of-N für lange Texte (>4000 chars).

    Pro Absatz separate humanize_bestofn-Runde. Final-Text = "\n\n".join(parts).
    Global-BGE-Sim wird am Ende auf den reassemblierten Text gemessen.
    """
    orig = text.strip()
    t0 = time.time()
    chunks = split_paragraphs(orig)

    if on_progress:
        on_progress(f"chunked: {len(chunks)} Absätze à Ø {sum(len(c) for c in chunks)//max(1,len(chunks))} chars")

    chunk_results = []
    final_parts = []
    total_cost = 0.0

    for i, chunk in enumerate(chunks):
        if on_progress:
            on_progress(f"  chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        if len(chunk) < 100:
            final_parts.append(chunk)
            chunk_results.append({"idx": i, "chars": len(chunk), "skipped": "too_short"})
            continue
        res = await humanize_bestofn(
            chunk, client=client,
            n_variants=n_variants_per_chunk,
            bge_threshold=bge_threshold,
            rank_by_pangram=rank_by_pangram,
            pangram_mock_cache=pangram_mock_cache,
        )
        final_parts.append(res.final_text)
        total_cost += res.total_cost_usd
        chunk_results.append({
            "idx": i, "chars": len(chunk),
            "out_chars": len(res.final_text),
            "bge_sim": round(res.bge_sim, 4),
            "pangram_post": res.pangram_post,
            "n_faithful": res.n_faithful,
            "stopped": res.stopped_reason,
        })

    final_text = "\n\n".join(final_parts)
    global_sim = (bge_similarity_batch(orig, [final_text])[0]
                  if final_text and final_text != orig else 1.0)

    return HumanizeBestofNResult(
        orig_text=orig,
        final_text=final_text,
        pangram_pre=None,
        pangram_post=None,
        bge_sim=global_sim,
        n_generated=sum(n_variants_per_chunk for c in chunks if len(c) >= 100),
        n_faithful=sum(r.get("n_faithful", 0) for r in chunk_results),
        total_cost_usd=round(total_cost, 4),
        duration_s=round(time.time() - t0, 1),
        stopped_reason="chunked_done",
        chunked=True,
        chunk_results=chunk_results,
    )
