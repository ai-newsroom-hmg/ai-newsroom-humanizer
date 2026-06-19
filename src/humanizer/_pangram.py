"""Pangram-Client (KI-Text-Erkennung).

Standard = Einzel-Tasks mit begrenzter Parallelität (PANGRAM_MODE=single):
    POST /task {text} → {task_id}; GET /task/<id> → stage=STAGE_SUCCESS +
    prediction_short/fraction_ai/fraction_ai_assisted/fraction_human (flach).
    Live verifiziert 2026-06-12: ~3 s pro Text.

PANGRAM_MODE=bulk nutzt die Bulk-API (POST /bulk → poll → /results, Rows
unter "items", Ergebnis je Row in "result"). Achtung: Die Bulk-Queue war im
Test extrem langsam (2 Texte >18 min in STAGE_INFERENCE) — nur für große
unbeaufsichtigte Läufe sinnvoll.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("ki-check.pangram")

BASE_URL = "https://text.external-api.pangram.com"
MODE = os.environ.get("PANGRAM_MODE", "single")  # single | bulk
CONCURRENCY = int(os.environ.get("PANGRAM_CONCURRENCY", "8"))
BULK_CHUNK = int(os.environ.get("PANGRAM_BULK_CHUNK", "50"))
POLL_INTERVAL = float(os.environ.get("PANGRAM_POLL_INTERVAL", "3"))
# Einzel-Task: Timeout pro Text. Bulk: Timeout pro Chunk.
TASK_TIMEOUT = float(os.environ.get("PANGRAM_TASK_TIMEOUT", "180"))
POLL_TIMEOUT = float(os.environ.get("PANGRAM_POLL_TIMEOUT", "3600"))

_FRACTION_KEYS = ("fraction_ai", "fraction_ai_assisted", "fraction_human")
_DONE_STAGES = ("stage_success",)
_FAIL_STAGES = ("stage_failed", "stage_error")


@dataclass
class PangramResult:
    item_id: str
    fraction_ai: Optional[float] = None
    fraction_ai_assisted: Optional[float] = None
    fraction_human: Optional[float] = None
    prediction: str = ""
    error: str = ""


def _find_fractions(obj) -> Optional[dict]:
    """Sucht rekursiv das Dict, das die fraction_*-Felder trägt."""
    if isinstance(obj, dict):
        if any(k in obj for k in _FRACTION_KEYS):
            return obj
        for v in obj.values():
            hit = _find_fractions(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_fractions(v)
            if hit:
                return hit
    return None


def parse_result(item_id: str, raw: dict) -> PangramResult:
    res = PangramResult(item_id=item_id)
    body = _find_fractions(raw)
    if not body:
        err = raw.get("error") or raw.get("message") or ""
        res.error = str(err) or "Kein Ergebnis im Response gefunden"
        return res
    res.fraction_ai = _as_float(body.get("fraction_ai"))
    res.fraction_ai_assisted = _as_float(body.get("fraction_ai_assisted"))
    res.fraction_human = _as_float(body.get("fraction_human"))
    res.prediction = str(body.get("prediction_short") or body.get("prediction") or "")
    if res.fraction_ai is None:
        # fraction-Keys vorhanden, aber Werte leer → als Fehler werten,
        # sonst entsteht ein "checked"-Record ohne Messwerte
        res.error = "Unvollständiges Pangram-Ergebnis (fraction_ai fehlt)"
    return res


def _as_float(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


class PangramClient:
    """Pangram API-Client.

    Modes:
    - Default (dev): live API-calls.
    - Staging: `mock_cache_path` set → load JSON cache, lookup per SHA256(text).
      Cache-Miss raises (kein silent live-fallback). UNBEDINGT-Skill 2 Live-Parity:
      Cache muss aus echten Live-Calls stammen.
    """

    def __init__(self, api_key: Optional[str] = None,
                 mock_cache_path: Optional[Path] = None):
        self._mock_cache: Optional[dict] = None
        self._mock_path = mock_cache_path
        if mock_cache_path:
            self._mock_cache = json.loads(Path(mock_cache_path).read_text(encoding="utf-8"))
            log.info("PangramClient: STAGING mode, %d cached entries from %s",
                     len(self._mock_cache), mock_cache_path)
            self._key = "STAGING_MOCK"
        else:
            self._key = api_key or os.environ["PANGRAM_API_KEY"]
        self._http: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        if self._mock_cache is None:
            self._http = httpx.AsyncClient(
                base_url=BASE_URL, timeout=60,
                headers={"x-api-key": self._key, "Content-Type": "application/json"})
        return self

    async def __aexit__(self, *a):
        if self._http:
            await self._http.aclose()

    async def check_bulk(self, items: list[dict], on_progress=None) -> dict[str, PangramResult]:
        """items = [{"id": …, "text": …}] → Ergebnisse je id."""
        if self._mock_cache is not None:
            return self._check_via_mock(items)
        if MODE == "bulk":
            return await self._check_via_bulk(items, on_progress)
        return await self._check_via_tasks(items, on_progress)

    def _check_via_mock(self, items: list[dict]) -> dict[str, PangramResult]:
        """Lookup each item by sha256(text) in mock_cache. Cache-Miss raises."""
        out: dict[str, PangramResult] = {}
        missing = []
        for it in items:
            sha = hashlib.sha256(it["text"].encode("utf-8")).hexdigest()
            entry = self._mock_cache.get(sha)
            if entry is None or "fraction_ai" not in entry:
                missing.append((it["id"], sha[:12]))
                continue
            res = PangramResult(
                item_id=it["id"],
                fraction_ai=entry.get("fraction_ai"),
                fraction_ai_assisted=entry.get("fraction_ai_assisted"),
                fraction_human=entry.get("fraction_human"),
                prediction=entry.get("prediction", ""),
            )
            out[it["id"]] = res
        if missing:
            raise RuntimeError(
                f"Pangram-MOCK Cache-Miss für {len(missing)} items "
                f"(IDs {[m[0] for m in missing[:3]]}…). "
                f"In staging-mode: kein silent live-fallback. "
                f"Cache path: {self._mock_path}. "
                f"Lösung: dev-mode für neue Texte verwenden, dann cache füllt sich; "
                f"oder explizit Test-Korpus auf gecachten Inputs einschränken."
            )
        return out

    # ---- Einzel-Tasks (Default) ------------------------------------------

    async def _check_via_tasks(self, items, on_progress) -> dict[str, PangramResult]:
        out: dict[str, PangramResult] = {}
        sem = asyncio.Semaphore(CONCURRENCY)
        done = 0

        async def one(it):
            nonlocal done
            async with sem:
                try:
                    out[it["id"]] = await self._run_task(it["id"], it["text"])
                except Exception as e:
                    log.exception("Pangram-Task %s fehlgeschlagen", it["id"])
                    out[it["id"]] = PangramResult(item_id=it["id"], error=str(e))
                done += 1
                if on_progress:
                    on_progress(done, len(items))

        await asyncio.gather(*(one(it) for it in items))
        return out

    async def _run_task(self, item_id: str, text: str) -> PangramResult:
        r = await self._http.post("/task", json={"text": text, "public_dashboard_link": False})
        r.raise_for_status()
        d = r.json()
        if _find_fractions(d):  # synchron beantwortet
            return parse_result(item_id, d)
        task_id = d.get("task_id") or d.get("id")
        if not task_id:
            raise RuntimeError(f"Kein task_id im Response: {str(d)[:300]}")
        waited = 0.0
        while waited <= TASK_TIMEOUT:
            await asyncio.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            r = await self._http.get(f"/task/{task_id}")
            r.raise_for_status()
            d = r.json()
            stage = str(d.get("stage") or d.get("status") or "").lower()
            if stage in _DONE_STAGES or _find_fractions(d):
                return parse_result(item_id, d)
            if stage in _FAIL_STAGES:
                raise RuntimeError(f"Pangram-Task fehlgeschlagen: {str(d)[:300]}")
        raise TimeoutError(f"Pangram-Task nicht fertig nach {TASK_TIMEOUT:.0f}s")

    # ---- Bulk-API (PANGRAM_MODE=bulk, sehr langsame Queue) ----------------

    async def _check_via_bulk(self, items, on_progress) -> dict[str, PangramResult]:
        out: dict[str, PangramResult] = {}
        for i in range(0, len(items), BULK_CHUNK):
            chunk = items[i:i + BULK_CHUNK]
            try:
                out.update(await self._run_chunk(chunk))
            except Exception as e:  # Chunk-Fehler nicht alles reißen lassen
                log.exception("Bulk-Chunk fehlgeschlagen")
                for it in chunk:
                    out.setdefault(it["id"], PangramResult(item_id=it["id"], error=str(e)))
            if on_progress:
                on_progress(min(i + BULK_CHUNK, len(items)), len(items))
        return out

    async def _run_chunk(self, chunk: list[dict]) -> dict[str, PangramResult]:
        r = await self._http.post("/bulk", json={"items": chunk})
        r.raise_for_status()
        sub = r.json()
        bulk_id = sub.get("bulk_id") or sub.get("id") or sub.get("bulkId")
        if not bulk_id:
            raise RuntimeError(f"Kein bulk_id im Submit-Response: {str(sub)[:300]}")

        waited = 0.0
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            r = await self._http.get(f"/bulk/{bulk_id}")
            r.raise_for_status()
            status = r.json()
            state = str(status.get("status") or status.get("state") or "").lower()
            total = status.get("total_items") or len(chunk)
            settled = (status.get("succeeded") or 0) + (status.get("failed") or 0)
            if (state in ("completed", "complete", "done", "finished", "success", "succeeded")
                    or status.get("completed_at") or settled >= total):
                break
            if state in ("failed", "error"):
                raise RuntimeError(f"Pangram-Bulk {bulk_id} fehlgeschlagen: {str(status)[:300]}")
            if waited > POLL_TIMEOUT:
                raise TimeoutError(f"Pangram-Bulk {bulk_id} nicht fertig nach {POLL_TIMEOUT:.0f}s")

        results: dict[str, PangramResult] = {}
        offset, limit = 0, 100
        while True:
            r = await self._http.get(f"/bulk/{bulk_id}/results",
                                     params={"offset": offset, "limit": limit})
            r.raise_for_status()
            page = r.json()
            rows = (page.get("items") or page.get("results")) if isinstance(page, dict) else page
            rows = rows or []
            for row in rows:
                rid = str(row.get("id") or row.get("item_id") or "")
                if not rid:
                    continue
                if row.get("error"):
                    results[rid] = PangramResult(item_id=rid, error=str(row["error"]))
                else:
                    results[rid] = parse_result(rid, row)
            if len(rows) < limit:
                break
            offset += limit
        return results
