"""Smoke tests for SmartMoneyDetector signals using synthetic candle data."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.data.types import Candle
from src.detector.smart_money import SmartMoneyDetector


def make_candles(closes: list[float], volumes: list[float]) -> list[Candle]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(ts=base + timedelta(hours=i), open=c, high=c, low=c, close=c, volume=v)
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


def test_short_history_skips():
    det = SmartMoneyDetector()
    candles = make_candles([0.10] * 50, [100.0] * 50)
    result = det.detect(candles)
    assert not result.triggered and result.score == 0


def test_slow_grind_plus_vol_trend_triggers():
    n = 130
    closes = [0.10 + i * 0.001 for i in range(n)]   # slow upward grind
    volumes = [50.0 + i * 1.0 for i in range(n)]   # linear vol increase
    det = SmartMoneyDetector()
    result = det.detect(make_candles(closes, volumes))
    assert "slow_grind" in result.signals
    assert "vol_trend" in result.signals
    assert result.triggered
    assert result.score >= 2


def test_vol_spike_alone_does_not_trigger():
    n = 130
    closes = [0.20] * n
    volumes = [10.0] * (n - 5) + [200.0] * 5  # huge late spike
    det = SmartMoneyDetector()
    result = det.detect(make_candles(closes, volumes))
    assert "vol_spike" in result.signals
    assert result.score == 1
    assert not result.triggered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
