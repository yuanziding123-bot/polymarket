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
        return _extract_json(text)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extractor: strips markdown fences, then locates the first
    `{ … }` block (or `[ … ]`) by bracket counting so trailing prose / leading
    preambles don't break parsing."""
    s = text.strip()
    if s.startswith("```"):
        # Drop opening fence and optional "json" tag
        s = s.lstrip("`").lstrip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        # Drop closing fence if still present
        if "```" in s:
            s = s.split("```", 1)[0]

    # Find the first balanced { ... } block
    start = s.find("{")
    if start == -1:
        log.warning(f"LLM response had no JSON object: {text[:200]}")
        return None
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        log.warning(f"LLM response had unbalanced JSON: {text[:200]}")
        return None
    try:
        return json.loads(s[start:end])
    except json.JSONDecodeError as exc:
        log.warning(f"LLM JSON parse failed ({exc}): {s[start:end][:200]}")
        return None
