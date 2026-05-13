"""Survey: run LLM probability estimate on every filtered candidate to see
the edge distribution. Helps answer 'is the system always-skip or sometimes-buy?'.

Run with: python -m tests._demo_edge_distribution
"""
from __future__ import annotations

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
    scanner = MarketScanner(client, FilterConfig())

    candidates = scanner.scan(raw_limit=500)
    print(f"\nProbing {len(candidates)} candidates with LLM…\n")

    # Synthetic 'best-combo' detection so we always run the LLM path
    fake = DetectionResult(triggered=True, score=2, signals=["breakout", "narrow_pullback"])

    rows = []
    for m in candidates:
        prob = estimator.estimate(m, fake)
        edge = prob.p_true - m.price
        rows.append((m, prob, edge))

    # Sort by edge descending so top opportunities surface first
    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"{'edge':>8}  {'mkt':>6}  {'p_true':>6}  {'conf':>6}  question")
    print("-" * 100)
    for m, prob, edge in rows:
        marker = "✓" if edge >= 0.06 and prob.confidence != "low" else " "
        q = m.question[:60]
        print(f"{edge:>+8.4f}  {m.price:>6.3f}  {prob.p_true:>6.3f}  "
              f"{prob.confidence:>6}  {marker} {q}")

    n_buys = sum(1 for _, p, e in rows if e >= 0.06 and p.confidence != "low")
    print(f"\n→ {n_buys}/{len(rows)} would clear the 6% edge floor")
    if rows:
        edges = [e for _, _, e in rows]
        print(f"  edge range: {min(edges):+.4f} to {max(edges):+.4f}, "
              f"median {sorted(edges)[len(edges)//2]:+.4f}")


if __name__ == "__main__":
    main()
