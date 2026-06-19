"""Tests für PangramClient staging-mode (mock cache)."""
from __future__ import annotations

import pytest

from humanizer._pangram import PangramClient


def test_mock_cache_hit_returns_cached(staging_cache_path, staging_docs):
    """Originale aus docs.jsonl sind im mini_pangram_cache.json drin → Hit."""
    pc = PangramClient(mock_cache_path=staging_cache_path)
    items = [{"id": d["label"], "text": d["volltext"]} for d in staging_docs]
    # Sync-call works because _check_via_mock is sync
    res = pc._check_via_mock(items)
    assert set(res.keys()) == {d["label"] for d in staging_docs}
    for label, r in res.items():
        assert r.fraction_ai is not None, f"{label} returned no fraction_ai"
        assert r.fraction_ai == 1.0, f"{label}: orig should be Pangram-flagged AI"


def test_mock_cache_miss_raises(staging_cache_path):
    pc = PangramClient(mock_cache_path=staging_cache_path)
    with pytest.raises(RuntimeError, match="Cache-Miss"):
        pc._check_via_mock([{"id": "novel", "text": "Ein ganz neuer Text der nicht im Cache ist."}])


def test_mock_init_does_not_need_api_key(staging_cache_path, monkeypatch):
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)
    pc = PangramClient(mock_cache_path=staging_cache_path)
    assert pc._mock_cache is not None
    assert pc._key == "STAGING_MOCK"


@pytest.mark.asyncio
async def test_check_bulk_routes_to_mock(staging_cache_path, staging_docs):
    """check_bulk in mock-mode runs without httpx + without API-key."""
    items = [{"id": d["label"], "text": d["volltext"]} for d in staging_docs]
    async with PangramClient(mock_cache_path=staging_cache_path) as pc:
        res = await pc.check_bulk(items)
    for d in staging_docs:
        assert d["label"] in res
        assert res[d["label"]].fraction_ai == 1.0
