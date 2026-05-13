"""Force the top-edge market through a full Bull/Bear/Judge debate, even if it's
below the production 6% floor — purely to inspect what the debate text looks like.

Temporarily monkey-patches EDGE_FLOOR. Do NOT use as a trading entry.
Run with: python -m tests._demo_full_debate
"""
from __future__ import annotations

import src.agents.debate as debate_mod
from src.data.news_client import NewsClient
from src.data.polymarket_client import PolymarketClient
from src.data.types import DetectionResult
from src.probability.estimator import ProbabilityEstimator
from src.probability.llm_client import LLMClient
from src.scanner.market_scanner import FilterConfig, MarketScanner


def main() -> None:
    debate_mod.EDGE_FLOOR = 0.01  # temp lowered for demo

    client = PolymarketClient()
    llm = LLMClient()
    news = NewsClient()
    estimator = ProbabilityEstimator(llm, news)
    debate = debate_mod.DebateOrchestrator(llm)
    scanner = MarketScanner(client, FilterConfig())

    candidates = scanner.scan(raw_limit=500)
    fake = DetectionResult(triggered=True, score=2, signals=["breakout", "narrow_pullback"])

    # Find the highest-edge candidate
    best = None
    best_edge = -1.0
    for m in candidates:
        prob = estimator.estimate(m, fake)
        edge = prob.p_true - m.price
        if edge > best_edge:
            best, best_edge, best_prob = m, edge, prob

    if best is None:
        print("no candidates")
        return

    print(f"\n=== Forcing full debate on best-edge market ===")
    print(f"Market: {best.question}")
    print(f"Outcome: {best.outcome}  Price: {best.price:.4f}  P_true: {best_prob.p_true:.4f}")
    print(f"Edge: {best_edge:+.4f}\n")

    news_items = news.search(best.question, max_results=5)
    decision = debate.run(best, fake, best_prob, news_items)

    print(f"FINAL DECISION: {decision.action}")
    print(f"  reason: {decision.reason}")
    print(f"  size: ${decision.position_size_usdc:.2f}")
    print(f"\nBULL summary: {decision.bull_summary}")
    print(f"\nBEAR summary: {decision.bear_summary}")


if __name__ == "__main__":
    main()
