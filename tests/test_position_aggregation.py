"""Tests for the new position-aggregation behaviour in TraceStore.upsert_position.

Before fix: a second buy on the same token silently overwrote nothing (only peak_price
was updated), losing the second $25.
After fix: the second buy merges into the existing open position with weight-averaged
entry price and summed USDC notional.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import config
from src.storage.db import TraceStore


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    db_path = tmp_path / "test_positions.db"
    monkeypatch.setattr(config.SETTINGS, "sqlite_path", db_path)
    return TraceStore(path=db_path)


def _make_pos(token_id, entry, size, opened_at=None):
    return SimpleNamespace(
        token_id=token_id,
        market_id=f"m_{token_id}",
        entry_price=entry,
        peak_price=entry,
        size_usdc=size,
        opened_at=opened_at or datetime.now(timezone.utc),
        expiry=datetime.now(timezone.utc) + timedelta(days=30),
    )


def test_single_position_inserts_normally(fresh_store):
    p = _make_pos("tok_A", 0.10, 25.0)
    fresh_store.upsert_position(p)
    rows = fresh_store.open_positions()
    assert len(rows) == 1
    assert rows[0]["entry_price"] == 0.10
    assert rows[0]["size_usdc"] == 25.0


def test_second_buy_same_price_doubles_size(fresh_store):
    p1 = _make_pos("tok_A", 0.10, 25.0)
    p2 = _make_pos("tok_A", 0.10, 25.0)
    fresh_store.upsert_position(p1)
    fresh_store.upsert_position(p2)
    rows = fresh_store.open_positions()
    assert len(rows) == 1
    assert rows[0]["size_usdc"] == 50.0
    assert abs(rows[0]["entry_price"] - 0.10) < 1e-9  # weighted avg of same price


def test_second_buy_different_price_weight_averages(fresh_store):
    # First: $25 @ $0.10 → 250 tokens
    # Second: $25 @ $0.20 → 125 tokens
    # Total: $50, 375 tokens, avg price = 50/375 = 0.1333
    p1 = _make_pos("tok_A", 0.10, 25.0)
    p2 = _make_pos("tok_A", 0.20, 25.0)
    fresh_store.upsert_position(p1)
    fresh_store.upsert_position(p2)
    rows = fresh_store.open_positions()
    assert len(rows) == 1
    assert rows[0]["size_usdc"] == 50.0
    assert abs(rows[0]["entry_price"] - (50.0 / 375.0)) < 1e-4


def test_peak_price_takes_max(fresh_store):
    p1 = _make_pos("tok_A", 0.10, 25.0)
    p1.peak_price = 0.12  # peak already higher than entry
    p2 = _make_pos("tok_A", 0.10, 25.0)
    p2.peak_price = 0.11
    fresh_store.upsert_position(p1)
    fresh_store.upsert_position(p2)
    rows = fresh_store.open_positions()
    assert rows[0]["peak_price"] == 0.12  # kept the higher one


def test_different_tokens_are_independent(fresh_store):
    fresh_store.upsert_position(_make_pos("tok_A", 0.10, 25.0))
    fresh_store.upsert_position(_make_pos("tok_B", 0.20, 30.0))
    rows = fresh_store.open_positions()
    assert len(rows) == 2


def test_reopen_after_close(fresh_store):
    # Close the first position, then a new buy on same token should be a fresh position.
    p1 = _make_pos("tok_A", 0.10, 25.0)
    fresh_store.upsert_position(p1)
    fresh_store.close_position("tok_A", reason="stop_loss", exit_price=0.05, pnl_usdc=-12.5)
    p2 = _make_pos("tok_A", 0.15, 30.0)
    fresh_store.upsert_position(p2)
    rows = fresh_store.open_positions()
    assert len(rows) == 1
    assert rows[0]["entry_price"] == 0.15
    assert rows[0]["size_usdc"] == 30.0
