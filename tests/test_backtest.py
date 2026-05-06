"""Backtest engine + metrics smoke tests using synthetic markets."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.backtest.engine import BacktestEngine, MarketUniverse
from src.backtest.metrics import aggregate
from src.data.types import Candle, Market


def _candles(closes: list[float], volumes: list[float]) -> list[Candle]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(ts=base + timedelta(hours=i), open=c, high=c, low=c, close=c, volume=v)
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


def _market(token_id: str = "tok_1") -> Market:
    return Market(
        market_id="m1", condition_id="c1",
        question="will event happen?", description="",
        outcome="YES", token_id=token_id, price=0.30,
        volume_24h=10_000.0, liquidity=20_000.0, spread=0.01,
        days_to_expiry=30.0,
        expiry=datetime(2026, 6, 1, tzinfo=timezone.utc), raw={},
    )


def test_engine_emits_trials_when_signals_fire():
    n = 200  # 120 history + 24 horizon + buffer
    closes = [0.10 + i * 0.001 for i in range(n)]      # slow_grind
    volumes = [50.0 + i * 1.0 for i in range(n)]      # vol_trend
    universe = MarketUniverse(
        markets=[_market()],
        candle_map={"tok_1": _candles(closes, volumes)},
    )
    engine = BacktestEngine(client=MagicMock())
    trials = engine.run(universe, horizon_bars=24, dedupe_bars=12)

    assert trials, "expected at least one trial when both slow_grind+vol_trend fire"
    for t in trials:
        assert "slow_grind" in t.signals or "vol_trend" in t.signals
        assert t.score >= 2
        assert t.exit_price > t.entry_price  # rising series → positive return


def test_metrics_overall_and_by_signal():
    n = 200
    closes = [0.10 + i * 0.001 for i in range(n)]
    volumes = [50.0 + i * 1.0 for i in range(n)]
    universe = MarketUniverse(markets=[_market()], candle_map={"tok_1": _candles(closes, volumes)})
    trials = BacktestEngine(client=MagicMock()).run(universe, horizon_bars=24, dedupe_bars=12)

    report = aggregate(trials)
    assert report.overall.n == len(trials)
    assert report.overall.hit_rate == 1.0  # monotonically rising series
    sig_labels = {s.label for s in report.by_signal}
    assert sig_labels  # at least one signal bucket
