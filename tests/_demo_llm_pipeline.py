"""Smoke-demo: force-trigger the full LLM-debate path on a real market.

Not a unit test — it makes real Polymarket + Anthropic API calls. Run with:
    python -m tests._demo_llm_pipeline
"""
from __future__ import annotations

from src.agents.debate import DebateOrchestrator
from src.data.news_client import NewsClient
from src.data.polymarket_client import PolymarketClient
from src.data.types import DetectionResult
from src.probability.estimator import ProbabilityEstimator
from src.probability.llm_client import LLMClient
from src.scanner.market_scanner import FilterConfig, MarketScanner


def main() -> None:
    client = PolymarketClient()
    llm = LLMClient()
    news = NewsClient()
    estimator = ProbabilityEstimator(llm, news)
    debate = DebateOrchestrator(llm)

    # Use the production filter so we get a candidate in the alpha sweet spot
    scanner = MarketScanner(client, FilterConfig())
    candidates = scanner.scan(raw_limit=500)
    if not candidates:
        print("no candidates")
        return
    market = candidates[0]
    print(f"\nMARKET: {market.question}")
    print(f"  price={market.price:.4f}  outcome={market.outcome}  vol24h={market.volume_24h:,.0f}")

    # Synthetic "perfect signal hit" — pretend the detector fired the winning combo
    detection = DetectionResult(triggered=True, score=2, signals=["breakout", "narrow_pullback"])

    print("\n[1/2] PROBABILITY ESTIMATE")
    prob = estimator.estimate(market, detection)
    print(f"  p_true={prob.p_true:.4f}  components={prob.components}")
    print(f"  reasoning={prob.reasoning}")
    print(f"  confidence={prob.confidence}  uncertainty={prob.uncertainty}")

    print("\n[2/2] DEBATE")
    news_items = news.search(market.question, max_results=5)
    decision = debate.run(market, detection, prob, news_items)
    print(f"  action: {decision.action}")
    print(f"  edge={decision.edge:+.4f}")
    print(f"  size_usdc=${decision.position_size_usdc:.2f}")
    print(f"  reason: {decision.reason}")
    print(f"  bull: {decision.bull_summary[:200]}")
    print(f"  bear: {decision.bear_summary[:200]}")


if __name__ == "__main__":
    main()
