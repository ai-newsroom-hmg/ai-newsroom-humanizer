"""Tests für humanizer.env.detect_env (dev/staging-Mode-Resolution)."""
from __future__ import annotations

import pytest

from humanizer.env import detect_env


def test_default_is_dev(monkeypatch):
    monkeypatch.delenv("PALIMPSEST_ENV", raising=False)
    env = detect_env()
    assert env.is_dev
    assert not env.pangram_cache_only


def test_env_var_staging(monkeypatch, staging_cache_path):
    monkeypatch.setenv("PALIMPSEST_ENV", "staging")
    monkeypatch.setenv("PALIMPSEST_PANGRAM_CACHE", str(staging_cache_path))
    env = detect_env()
    assert env.is_staging
    assert env.pangram_cache_only
    assert env.pangram_cache_path == staging_cache_path


def test_cli_flag_overrides_env_var(monkeypatch, staging_cache_path):
    monkeypatch.setenv("PALIMPSEST_ENV", "staging")
    monkeypatch.setenv("PALIMPSEST_PANGRAM_CACHE", str(staging_cache_path))
    env = detect_env(cli_flag="dev")
    assert env.is_dev


def test_invalid_env_raises(monkeypatch):
    monkeypatch.setenv("PALIMPSEST_ENV", "production")
    with pytest.raises(ValueError, match="dev.*staging"):
        detect_env()


def test_staging_without_cache_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PALIMPSEST_ENV", "staging")
    monkeypatch.setenv("PALIMPSEST_PANGRAM_CACHE", str(tmp_path / "nope.json"))
    with pytest.raises(RuntimeError, match="Pangram-Cache"):
        detect_env()


def test_ollama_url_threading(monkeypatch, staging_cache_path):
    monkeypatch.setenv("PALIMPSEST_ENV", "staging")
    monkeypatch.setenv("PALIMPSEST_PANGRAM_CACHE", str(staging_cache_path))
    monkeypatch.setenv("PALIMPSEST_OLLAMA_URL", "http://ruediger:11434")
    env = detect_env()
    assert env.ollama_url == "http://ruediger:11434"
