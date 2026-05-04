"""Module 3 — probability ensemble: P_true = w1·P_llm + w2·P_base + w3·P_news + w4·P_corr."""
from __future__ import annotations

from dataclasses import dataclass

from src.data.news_client import NewsClient, NewsItem
from src.data.types import DetectionResult, Market, ProbabilityEstimate
from src.probability.llm_client import LLMClient
from src.probability.prompts import PROBABILITY_PROMPT
from src.utils.logger import get_logger

log = get_logger("probability")


@dataclass(frozen=True)
class EnsembleWeights:
    llm: float = 0.40
    base_rate: float = 0.20
    news: float = 0.25
    correlation: float = 0.15

    def normalise(self) -> "EnsembleWeights":
        s = self.llm + self.base_rate + self.news + self.correlation
        if s <= 0:
            return self
        return EnsembleWeights(self.llm / s, self.base_rate / s, self.news / s, self.correlation / s)


# Coarse base-rate prior for low-probability binary markets in the 5%-50% band.
# Tuned from the design doc's case study (true 60-90% on contracts trading 7-19c)
# but kept conservative — the LLM and news components dominate.
_DEFAULT_BASE_RATE = 0.18


class ProbabilityEstimator:
    def __init__(
        self,
        llm: LLMClient,
        news: NewsClient,
        weights: EnsembleWeights | None = None,
    ) -> None:
        self._llm = llm
        self._news = news
        self._weights = (weights or EnsembleWeights()).normalise()

    def estimate(
        self,
        market: Market,
        detection: DetectionResult,
        correlated_price: float | None = None,
    ) -> ProbabilityEstimate:
        news_items = self._news.search(market.question, max_results=5)

        p_llm, reasoning, uncertainty, confidence = self._llm_component(market, detection, news_items)
        p_base = _DEFAULT_BASE_RATE
        p_news = self._news_component(market, news_items)
        p_corr = correlated_price if correlated_price is not None else market.price

        w = self._weights
        p_true = (
            w.llm * p_llm
            + w.base_rate * p_base
            + w.news * p_news
            + w.correlation * p_corr
        )
        p_true = max(0.0, min(1.0, p_true))

        components = {"llm": p_llm, "base_rate": p_base, "news": p_news, "corr": p_corr}
        log.debug(f"P_true={p_true:.3f} components={components}")
        return ProbabilityEstimate(
            p_true=p_true,
            components=components,
            reasoning=reasoning,
            confidence=confidence,
            uncertainty=uncertainty,
        )

    # ------------------------------------------------------------------

    def _llm_component(
        self,
        market: Market,
        detection: DetectionResult,
        news_items: list[NewsItem],
    ) -> tuple[float, list[str], str, str]:
        if not self._llm.is_ready():
            return market.price, ["LLM disabled — fell back to market price"], "no_llm", "low"

        prompt = PROBABILITY_PROMPT.format(
            question=market.question,
            description=market.description[:1500],
            expiry=market.expiry.isoformat(),
            days_to_expiry=market.days_to_expiry,
            outcome=market.outcome,
            price=market.price,
            signals=", ".join(detection.signals) or "none",
            n_news=len(news_items),
            news_block=_format_news(news_items),
        )
        result = self._llm.complete_json(prompt, max_tokens=600, temperature=0.2)
        if not result:
            return market.price, ["LLM call failed"], "llm_error", "low"

        try:
            p = float(result.get("probability", market.price))
            p = max(0.0, min(1.0, p))
        except (TypeError, ValueError):
            p = market.price
        reasoning = list(result.get("reasoning") or [])
        uncertainty = str(result.get("uncertainty") or "")
        confidence = str(result.get("confidence") or "low").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        return p, reasoning, uncertainty, confidence

    @staticmethod
    def _news_component(market: Market, news_items: list[NewsItem]) -> float:
        """Naive sentiment-free heuristic: more recent/relevant items lift prob slightly.

        Good enough for v1; the LLM component carries the real news intelligence.
        """
        if not news_items:
            return market.price
        bump = min(0.05, 0.01 * len(news_items))
        return max(0.0, min(1.0, market.price + bump))


def _format_news(items: list[NewsItem]) -> str:
    if not items:
        return "(no news available)"
    return "\n".join(f"- [{i.published or 'n/a'}] {i.title} — {i.snippet[:200]}" for i in items)
