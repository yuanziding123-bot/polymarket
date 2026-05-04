"""Prompt templates for probability estimation and Bull/Bear/Risk debate."""
from __future__ import annotations

PROBABILITY_PROMPT = """\
You analyse Polymarket prediction markets.

Market: {question}
Description: {description}
Expiry (UTC): {expiry}
Days to expiry: {days_to_expiry:.1f}
Outcome side under analysis: {outcome}
Current market price (implied prob): {price:.4f}
Detected smart-money signals: {signals}

Recent news context (top {n_news}):
{news_block}

Tasks:
1. Estimate the true probability of this outcome (0.0–1.0).
2. List up to 3 key reasons.
3. Identify the dominant uncertainty.
4. Confidence: low | medium | high.

Respond ONLY with JSON:
{{"probability": 0.xx, "reasoning": ["...","..."], "uncertainty": "...", "confidence": "low|medium|high"}}
"""

BULL_PROMPT = """\
You are the BULL agent: build the strongest case that the outcome WILL occur and that
buying at the current price is profitable.

Market: {question}
Outcome: {outcome}
Price: {price:.4f} | Estimated true prob: {p_true:.4f} | Edge: {edge:+.4f}
Smart-money signals: {signals}
News context:
{news_block}

Respond JSON:
{{"thesis": "...", "key_evidence": ["...","..."], "risks_acknowledged": ["..."], "conviction": 0.0-1.0}}
"""

BEAR_PROMPT = """\
You are the BEAR agent: build the strongest case AGAINST buying. Look for crowded
positioning, news priced in, expiry mechanics, illiquidity, regulatory traps.

Market: {question}
Outcome: {outcome}
Price: {price:.4f} | Estimated true prob: {p_true:.4f} | Edge: {edge:+.4f}
Smart-money signals: {signals}
News context:
{news_block}

Respond JSON:
{{"thesis": "...", "key_evidence": ["...","..."], "bullish_points_addressed": ["..."], "conviction": 0.0-1.0}}
"""

JUDGE_PROMPT = """\
You are the RISK JUDGE. Weigh the bull and bear cases and produce a final verdict.

Market: {question}
Price: {price:.4f} | True prob estimate: {p_true:.4f} | Edge: {edge:+.4f}
Bull conviction: {bull_conv:.2f} — {bull_thesis}
Bear conviction: {bear_conv:.2f} — {bear_thesis}

Rules:
- Recommend BUY only if edge > 0.06 AND bull conviction clearly exceeds bear AND
  no fatal risk identified.
- Otherwise recommend SKIP.
- size_multiplier scales the Kelly position (0.0 skip, up to 1.0 full conservative Kelly).

Respond JSON:
{{"action": "buy|skip", "size_multiplier": 0.0-1.0, "reason": "...",
  "bull_summary": "...", "bear_summary": "..."}}
"""
