"""Minimaler OpenRouter-Client fuer die news-wire-Test-Pipeline (Sonnet 4.x).

Key kommt aus ~/.config/openrouter/key (analog news-wire/llm.py).
"""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_DEFAULT = "anthropic/claude-sonnet-4.5"
KEY_FILE = Path.home() / ".config" / "openrouter" / "key"


def _key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    raise RuntimeError("OPENROUTER_API_KEY env oder ~/.config/openrouter/key fehlt")


class ORClient:
    def __init__(self, model: str = MODEL_DEFAULT):
        self._headers = {
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-newsroom-hmg/ki-check",
            "X-Title": "ki-check-test-corpus",
        }
        self.model = model

    async def complete(
        self, system: str, user: str,
        *, temperature: float = 0.4, max_tokens: int = 2048, retries: int = 4,
    ) -> dict:
        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=600) as http:
            for attempt in range(retries):
                try:
                    r = await http.post(API_URL, headers=self._headers, json=body)
                    if r.status_code in (400, 401, 403):
                        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")
                    if r.status_code == 429 or r.status_code >= 500:
                        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                    data = r.json()
                    usage = data.get("usage") or {}
                    return {
                        "text": data["choices"][0]["message"]["content"],
                        "input_tokens": int(usage.get("prompt_tokens", 0)),
                        "output_tokens": int(usage.get("completion_tokens", 0)),
                        "cost_usd": float(usage.get("cost") or 0.0),
                    }
                except (httpx.HTTPError, httpx.HTTPStatusError) as e:
                    last_err = e
                    if attempt == retries - 1:
                        raise
                    backoff = (2 ** attempt) + random.random()
                    await asyncio.sleep(backoff)
        raise last_err or RuntimeError("complete failed")
