"""Tests for trade -> hourly volume bucketing."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.data.types import Candle
from src.data.volume import enrich_candles_with_volume


def _candles(n: int, start_ts: datetime, bar_seconds: int = 3600) -> list[Candle]:
    return [
        Candle(
            ts=start_ts + timedelta(seconds=i * bar_seconds),
            open=0.30, high=0.30, low=0.30, close=0.30, volume=0.0,
        )
        for i in range(n)
    ]


def test_enrich_with_no_trades_keeps_zero_volume():
    client = MagicMock()
    client.fetch_market_trades.return_value = []
    candles = _candles(5, datetime(2026, 1, 1, tzinfo=timezone.utc))
    out = enrich_candles_with_volume(candles, "cond_1", "tok_1", client, cache=None)
    assert all(c.volume == 0.0 for c in out)


def test_enrich_buckets_by_hour():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = _candles(3, start)  # 3 hourly bars 00:00, 01:00, 02:00

    # 3 trades: bar 0 gets 100, bar 1 gets 0, bar 2 gets 50+25=75
    trades = [
        {"asset": "tok_1", "timestamp": int(start.timestamp()) + 600, "size": 100.0},   # 00:10
        {"asset": "tok_1", "timestamp": int(start.timestamp()) + 7300, "size": 50.0},   # 02:01
        {"asset": "tok_1", "timestamp": int(start.timestamp()) + 7800, "size": 25.0},   # 02:10
        {"asset": "OTHER", "timestamp": int(start.timestamp()) + 1200, "size": 999.0},  # wrong asset, ignored
    ]
    client = MagicMock()
    client.fetch_market_trades.return_value = trades

    out = enrich_candles_with_volume(candles, "cond_1", "tok_1", client, cache=None)
    assert out[0].volume == 100.0
    assert out[1].volume == 0.0
    assert out[2].volume == 75.0


def test_trades_outside_window_are_ignored():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = _candles(2, start)

    trades = [
        {"asset": "tok_1", "timestamp": int(start.timestamp()) - 100, "size": 999.0},   # before
        {"asset": "tok_1", "timestamp": int(start.timestamp()) + 8000, "size": 999.0},  # after bar[1]
        {"asset": "tok_1", "timestamp": int(start.timestamp()) + 1500, "size": 42.0},   # bar 0
    ]
    client = MagicMock()
    client.fetch_market_trades.return_value = trades

    out = enrich_candles_with_volume(candles, "cond_1", "tok_1", client, cache=None)
    assert out[0].volume == 42.0
    assert out[1].volume == 0.0
