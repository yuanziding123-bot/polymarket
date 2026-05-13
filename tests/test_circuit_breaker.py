"""CircuitBreaker rules — each tested in isolation against a fresh in-memory store."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import config
from src.risk.circuit_breaker import CircuitBreaker
from src.storage.db import TraceStore


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    # Point SQLite at a tmp path so tests don't touch the real db
    db_path = tmp_path / "test_traces.db"
    monkeypatch.setattr(config.SETTINGS, "sqlite_path", db_path)
    return TraceStore(path=db_path)


def _insert_closed_position(store: TraceStore, token_id: str, pnl_usdc: float,
                            closed_at: datetime | None = None,
                            size_usdc: float = 100.0) -> None:
    closed_at = closed_at or datetime.now(timezone.utc)
    opened = closed_at - timedelta(days=1)
    expiry = closed_at + timedelta(days=7)
    with store._conn() as cx:  # noqa: SLF001
        cx.execute(
            """INSERT INTO positions
               (token_id, market_id, entry_price, peak_price, size_usdc,
                opened_at, expiry, closed_at, close_reason, exit_price, pnl_usdc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (token_id, f"m_{token_id}", 0.15, 0.18, size_usdc,
             opened.isoformat(), expiry.isoformat(),
             closed_at.isoformat(), "stop_loss", 0.10, pnl_usdc),
        )


def _insert_open_position(store: TraceStore, token_id: str, size_usdc: float = 100.0) -> None:
    now = datetime.now(timezone.utc)
    with store._conn() as cx:  # noqa: SLF001
        cx.execute(
            """INSERT INTO positions
               (token_id, market_id, entry_price, peak_price, size_usdc, opened_at, expiry)
               VALUES (?,?,?,?,?,?,?)""",
            (token_id, f"m_{token_id}", 0.15, 0.15, size_usdc,
             now.isoformat(), (now + timedelta(days=7)).isoformat()),
        )


# ----- daily loss rule --------------------------------------------------

def test_passes_when_no_history(fresh_store):
    cb = CircuitBreaker(fresh_store, bankroll_usdc=1000.0)
    assert cb.check(10.0).allowed


def test_blocks_when_daily_loss_exceeds_limit(fresh_store, monkeypatch):
    monkeypatch.setattr(config.SETTINGS, "max_daily_loss_pct", 0.05)
    # bankroll 1000 * 5% = 50 loss limit. Insert a -60 loss today.
    _insert_closed_position(fresh_store, "tok_loss", pnl_usdc=-60.0)
    cb = CircuitBreaker(fresh_store, bankroll_usdc=1000.0)
    v = cb.check(10.0)
    assert not v.allowed
    assert "daily loss" in v.reason


def test_does_not_block_when_yesterday_losses(fresh_store):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1, hours=2)
    _insert_closed_position(fresh_store, "tok_yest", pnl_usdc=-500.0, closed_at=yesterday)
    cb = CircuitBreaker(fresh_store, bankroll_usdc=1000.0)
    assert cb.check(10.0).allowed


# ----- exposure rule ----------------------------------------------------

def test_blocks_when_total_exposure_exceeds_cap(fresh_store, monkeypatch):
    monkeypatch.setattr(config.SETTINGS, "max_total_exposure_pct", 0.50)
    # cap 1000 * 50% = 500. Open 400 + prospective 200 = 600 → block.
    _insert_open_position(fresh_store, "tok_a", size_usdc=400.0)
    cb = CircuitBreaker(fresh_store, bankroll_usdc=1000.0)
    v = cb.check(prospective_size_usdc=200.0)
    assert not v.allowed
    assert "exposure" in v.reason


def test_allows_when_within_exposure_cap(fresh_store):
    _insert_open_position(fresh_store, "tok_a", size_usdc=100.0)
    cb = CircuitBreaker(fresh_store, bankroll_usdc=1000.0)
    assert cb.check(50.0).allowed


# ----- concurrent positions rule ----------------------------------------

def test_blocks_when_concurrent_cap_hit(fresh_store, monkeypatch):
    monkeypatch.setattr(config.SETTINGS, "max_concurrent_positions", 3)
    for i in range(3):
        _insert_open_position(fresh_store, f"tok_{i}", size_usdc=10.0)
    cb = CircuitBreaker(fresh_store, bankroll_usdc=10_000.0)
    v = cb.check(10.0)
    assert not v.allowed
    assert "open positions" in v.reason


# ----- consecutive losses rule ------------------------------------------

def test_blocks_after_n_consecutive_losses(fresh_store, monkeypatch):
    monkeypatch.setattr(config.SETTINGS, "max_consecutive_losses", 3)
    monkeypatch.setattr(config.SETTINGS, "consecutive_loss_cooldown_seconds", 3600)
    base = datetime.now(timezone.utc) - timedelta(minutes=10)
    for i in range(3):
        _insert_closed_position(fresh_store, f"tok_l{i}", pnl_usdc=-5.0,
                                closed_at=base + timedelta(seconds=i))
    cb = CircuitBreaker(fresh_store, bankroll_usdc=10_000.0)
    v = cb.check(10.0)
    assert not v.allowed
    assert "cooldown" in v.reason


def test_cooldown_lifts_after_expiry(fresh_store, monkeypatch):
    monkeypatch.setattr(config.SETTINGS, "max_consecutive_losses", 3)
    monkeypatch.setattr(config.SETTINGS, "consecutive_loss_cooldown_seconds", 60)
    long_ago = datetime.now(timezone.utc) - timedelta(hours=2)
    for i in range(3):
        _insert_closed_position(fresh_store, f"tok_l{i}", pnl_usdc=-5.0,
                                closed_at=long_ago + timedelta(seconds=i))
    cb = CircuitBreaker(fresh_store, bankroll_usdc=10_000.0)
    assert cb.check(10.0).allowed


def test_does_not_block_when_mixed_pnls(fresh_store, monkeypatch):
    monkeypatch.setattr(config.SETTINGS, "max_consecutive_losses", 3)
    base = datetime.now(timezone.utc) - timedelta(minutes=10)
    # two losses then a win → not all-N-losses
    _insert_closed_position(fresh_store, "tok_w", pnl_usdc=+5.0, closed_at=base + timedelta(seconds=2))
    _insert_closed_position(fresh_store, "tok_l1", pnl_usdc=-5.0, closed_at=base + timedelta(seconds=1))
    _insert_closed_position(fresh_store, "tok_l2", pnl_usdc=-5.0, closed_at=base)
    cb = CircuitBreaker(fresh_store, bankroll_usdc=10_000.0)
    assert cb.check(10.0).allowed
