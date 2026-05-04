"""Thin Anthropic SDK wrapper with prompt caching enabled by default."""
from __future__ import annotations

import json
from typing import Any

from config import SETTINGS
from src.utils.logger import get_logger

log = get_logger("llm")

_SYSTEM_BASE = (
    "You are a quantitative analyst specialised in prediction-market pricing. "
    "Always reply with strict JSON matching the requested schema; never wrap in markdown."
)


class LLMClient:
    def __init__(self) -> None:
        self._client = None
        if SETTINGS.anthropic_api_key:
            try:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=SETTINGS.anthropic_api_key)
            except ImportError:
                log.warning("anthropic SDK not installed; LLM disabled.")
        else:
            log.info("ANTHROPIC_API_KEY not set; LLM calls will return None.")

    def is_ready(self) -> bool:
        return self._client is not None

    def complete_json(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> dict[str, Any] | None:
        """Send a single-turn prompt and parse the response as JSON.

        Caches the system block (1h TTL) to avoid re-billing for shared instructions.
        """
        if not self._client:
            return None

        system_blocks = [
            {
                "type": "text",
                "text": system_prompt or _SYSTEM_BASE,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        try:
            resp = self._client.messages.create(
                model=SETTINGS.claude_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_blocks,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            log.warning(f"Anthropic call failed: {exc}")
            return None

        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning(f"LLM response was not JSON: {text[:200]}")
            return None
