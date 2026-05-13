"""Module 4 — Bull / Bear / Risk-Judge debate over a candidate market."""
from __future__ import annotations

from dataclasses import dataclass

from config import SETTINGS
from src.data.news_client import NewsItem
from src.data.types import DetectionResult, Market, ProbabilityEstimate, TradeDecision
from src.probability.llm_client import LLMClient
from src.probability.prompts import BEAR_PROMPT, BULL_PROMPT, JUDGE_PROMPT
from src.utils.kelly import kelly_position_usdc
from src.utils.logger import get_logger

log = get_logger("debate")

EDGE_FLOOR = 0.06  # design doc strategy A floor


@dataclass
class DebateOutput:
    bull: dict
    bear: dict
    judge: dict


class DebateOrchestrator:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(
        self,
        market: Market,
        detection: DetectionResult,
        prob: ProbabilityEstimate,
        news_items: list[NewsItem],
        bankroll: float = SETTINGS.bankroll_usdc,
    ) -> TradeDecision:
        edge = prob.p_true - market.price
        ctx = {
            "question": market.question,
            "outcome": market.outcome,
            "price": market.price,
            "p_true": prob.p_true,
            "edge": edge,
            "signals": ", ".join(detection.signals) or "none",
            "news_block": _format_news(news_items),
        }

        if edge < EDGE_FLOOR or prob.confidence == "low":
            return self._skip(market, prob, edge, reason=f"edge<{EDGE_FLOOR} or low confidence")

        bull = self._llm.complete_json(BULL_PROMPT.format(**ctx), max_tokens=900, temperature=0.4) or {}
        bear = self._llm.complete_json(BEAR_PROMPT.format(**ctx), max_tokens=900, temperature=0.4) or {}

        judge_prompt = JUDGE_PROMPT.format(
            question=market.question,
            price=market.price,
            p_true=prob.p_true,
            edge=edge,
            bull_conv=float(bull.get("conviction") or 0.0),
            bull_thesis=str(bull.get("thesis") or "n/a"),
            bear_conv=float(bear.get("conviction") or 0.0),
            bear_thesis=str(bear.get("thesis") or "n/a"),
        )
        judge = self._llm.complete_json(judge_prompt, max_tokens=400, temperature=0.2) or {}

        action = str(judge.get("action") or "skip").lower()
        if action != "buy":
            return self._skip(market, prob, edge, reason=str(judge.get("reason") or "judge_skip"),
                              bull=bull, bear=bear)

        size_mult = float(judge.get("size_multiplier") or 0.0)
        size_mult = max(0.0, min(1.0, size_mult))
        kelly_usdc = kelly_position_usdc(
            p_true=prob.p_true,
            p_market=market.price,
            bankroll=bankroll,
            fraction_multiplier=0.25 * size_mult if size_mult else 0.25,
            max_fraction=SETTINGS.max_position_fraction,
        )
        if kelly_usdc <= 0:
            return self._skip(market, prob, edge, reason="kelly=0", bull=bull, bear=bear)

        return TradeDecision(
            market_id=market.market_id,
            token_id=market.token_id,
            side="buy",
            market_price=market.price,
            p_true=prob.p_true,
            edge=edge,
            position_size_usdc=kelly_usdc,
            action="buy",
            reason=str(judge.get("reason") or ""),
            bull_summary=str(judge.get("bull_summary") or bull.get("thesis") or ""),
            bear_summary=str(judge.get("bear_summary") or bear.get("thesis") or ""),
        )

    @staticmethod
    def _skip(
        market: Market,
        prob: ProbabilityEstimate,
        edge: float,
        reason: str,
        bull: dict | None = None,
        bear: dict | None = None,
    ) -> TradeDecision:
        return TradeDecision(
            market_id=market.market_id,
            token_id=market.token_id,
            side="buy",
            market_price=market.price,
            p_true=prob.p_true,
            edge=edge,
            position_size_usdc=0.0,
            action="skip",
            reason=reason,
            bull_summary=str((bull or {}).get("thesis") or ""),
            bear_summary=str((bear or {}).get("thesis") or ""),
        )


def _format_news(items: list[NewsItem]) -> str:
    if not items:
        return "(no news)"
    return "\n".join(f"- {i.title}: {i.snippet[:160]}" for i in items[:5])
