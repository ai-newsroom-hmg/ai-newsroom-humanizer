"""Pytest conftest: project-root resolution, fixtures für staging-tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STAGING = ROOT / "tests" / "staging_corpus"


@pytest.fixture
def staging_cache_path() -> Path:
    p = STAGING / "mini_pangram_cache.json"
    assert p.exists(), f"staging mini cache missing: {p}"
    return p


@pytest.fixture
def staging_docs() -> list[dict]:
    p = STAGING / "docs.jsonl"
    if not p.exists():
        pytest.skip("docs.jsonl mit Volltexten ist nicht im public-Repo (Tagesspiegel-Copyright). "
                    "Lokal: rsync von ruediger oder selbst generieren.")
    return [json.loads(l) for l in p.read_text().split("\n") if l.strip()]


@pytest.fixture
def staging_expected() -> dict:
    return json.loads((STAGING / "expected_results.json").read_text())


@pytest.fixture
def staging_env(monkeypatch, staging_cache_path):
    """Set up PALIMPSEST_ENV=staging + cache path."""
    monkeypatch.setenv("PALIMPSEST_ENV", "staging")
    monkeypatch.setenv("PALIMPSEST_PANGRAM_CACHE", str(staging_cache_path))
    yield
