"""Detector diagnostic: for every current live candidate, fetch full history +
volume and report which of the 5 signals fire. Shows whether the AND-whitelist
is the bottleneck or the individual thresholds are.

Run with: python -m tests._demo_detector_diag
"""
from __future__ import annotations

from collections import Counter

from src.data.polymarket_client import PolymarketClient
from src.data.volume import enrich_candles_with_volume
from src.detector.smart_money import SmartMoneyDetector
from src.scanner.market_scanner import FilterConfig, MarketScanner
from src.storage.db import TraceStore


def main() -> None:
    client = PolymarketClient()
    store = TraceStore()
    scanner = MarketScanner(client, FilterConfig())
    detector = SmartMoneyDetector()

    candidates = scanner.scan(raw_limit=1000)
    print(f"\nDetector diagnostic on {len(candidates)} live candidates")
    print(f"{'price':>6}  {'days':>5}  {'signals':<45}  question")
    print("-" * 130)

    sig_counter: Counter = Counter()
    score_counter: Counter = Counter()
    triggered_count = 0
    no_signal_count = 0

    for m in candidates:
        candles = client.fetch_price_history(m.token_id, interval="1m", fidelity=60)
        candles = enrich_candles_with_volume(candles, m.condition_id, m.token_id, client, store)
        result = detector.detect(candles)

        sig_str = "+".join(result.signals) if result.signals else "(none)"
        triggered_marker = "*" if result.triggered else " "
        print(f"{m.price:>6.3f}  {m.days_to_expiry:>5.1f}  {triggered_marker} score={result.score} "
              f"{sig_str:<40}  {m.question[:60]}")

        if result.triggered:
            triggered_count += 1
        if not result.signals:
            no_signal_count += 1
        for s in result.signals:
            sig_counter[s] += 1
        score_counter[result.score] += 1

    print(f"\nSummary on {len(candidates)} markets:")
    print(f"  triggered (score ≥ 2):  {triggered_count}")
    print(f"  no signal at all:       {no_signal_count}")
    print(f"\n  individual signal hits:")
    for s, n in sig_counter.most_common():
        print(f"    {s:<18}: {n}")
    print(f"\n  score distribution:")
    for k in sorted(score_counter.keys()):
        print(f"    score={k}: {score_counter[k]}")


if __name__ == "__main__":
    main()
