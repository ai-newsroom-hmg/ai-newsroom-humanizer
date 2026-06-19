"""Palimpsest Environment-Handling (dev / staging).

Modi:
- **dev** (Default): live OpenRouter (Mistral-3.2-instruct) + live Pangram-API.
  Echte Performance, echte Kosten (~$1-5 pro Lauf).

- **staging**: deterministische Mocks für Regression-Tests / CI.
  - Pangram-Antworten aus pangram_cache.json (Cache aus echten Live-Calls,
    UNBEDINGT-Skill 2 Live-Parity erfüllt).
  - OpenRouter wahlweise via Ollama-URL (z.B. ruediger:11434) wenn
    PALIMPSEST_OLLAMA_URL gesetzt — sonst live OpenRouter.
  - Test-Korpus-Subset (3 Docs) für reproduzierbare Regression.

Selektion via:
- `--env dev|staging` Flag
- `PALIMPSEST_ENV` Env-Variable
- Default: dev

Lookup-Priorität: CLI-Flag > Env-Var > Default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PalimpsestEnv:
    name: str  # "dev" | "staging"
    pangram_cache_only: bool  # True: nur Cache, kein API-Call (Pflicht in staging)
    pangram_cache_path: Optional[Path]
    ollama_url: Optional[str]  # If set: route Mistral calls through Ollama instead of OpenRouter
    test_corpus_path: Optional[Path]

    @property
    def is_dev(self) -> bool:
        return self.name == "dev"

    @property
    def is_staging(self) -> bool:
        return self.name == "staging"


def detect_env(cli_flag: Optional[str] = None) -> PalimpsestEnv:
    """Resolve current env. CLI-flag wins over env-var wins over default 'dev'."""
    name = cli_flag or os.environ.get("PALIMPSEST_ENV", "dev")
    if name not in ("dev", "staging"):
        raise ValueError(f"PALIMPSEST_ENV must be 'dev' or 'staging', got: {name!r}")

    ollama_url = os.environ.get("PALIMPSEST_OLLAMA_URL")
    cache_default = PROJECT_ROOT / "data" / "phase2" / "pangram_cache.json"
    cache_path = Path(os.environ.get("PALIMPSEST_PANGRAM_CACHE", str(cache_default)))
    corpus_default = PROJECT_ROOT / "tests" / "staging_corpus"
    corpus_path = Path(os.environ.get("PALIMPSEST_TEST_CORPUS", str(corpus_default)))

    if name == "staging":
        if not cache_path.exists():
            raise RuntimeError(
                f"Staging-Mode braucht Pangram-Cache, fehlt: {cache_path}. "
                f"Setze PALIMPSEST_PANGRAM_CACHE oder kopier den Cache aus phase2/."
            )
        return PalimpsestEnv(
            name="staging",
            pangram_cache_only=True,
            pangram_cache_path=cache_path,
            ollama_url=ollama_url,
            test_corpus_path=corpus_path if corpus_path.exists() else None,
        )

    return PalimpsestEnv(
        name="dev",
        pangram_cache_only=False,
        pangram_cache_path=cache_path if cache_path.exists() else None,
        ollama_url=ollama_url,
        test_corpus_path=None,
    )
