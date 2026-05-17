"""Test TraceStore.recent_signal_within() — the 12h dedupe used by the live pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import config
from src.storage.db import TraceStore


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(config.SETTINGS, "sqlite_path", db_path)
    return TraceStore(path=db_path)


def _fake_market(market_id: str = "m1", token_id: str = "tok_1") -> SimpleNamespace:
    return SimpleNamespace(
        market_id=market_id, token_id=token_id,
        question="will event happen?", price=0.20,
    )


def _insert_signal_at(store: TraceStore, market_id: str, when: datetime) -> None:
    with store._conn() as cx:  # noqa: SLF001
        cx.execute(
            "INSERT INTO signals(ts,market_id,token_id,question,price,signals,score) "
            "VALUES(?,?,?,?,?,?,?)",
            (when.isoformat(), market_id, "tok", "q", 0.2,
             "narrow_pullback,breakout", 2),
        )


def test_returns_false_when_no_history(fresh_store):
    assert not fresh_store.recent_signal_within("m_new", hours=12.0)


def test_returns_true_for_recent_signal(fresh_store):
    _insert_signal_at(fresh_store, "m_recent", datetime.now(timezone.utc) - timedelta(hours=2))
    assert fresh_store.recent_signal_within("m_recent", hours=12.0)


def test_returns_false_for_signal_outside_window(fresh_store):
    _insert_signal_at(fresh_store, "m_old", datetime.now(timezone.utc) - timedelta(hours=13))
    assert not fresh_store.recent_signal_within("m_old", hours=12.0)


def test_isolates_per_market(fresh_store):
    _insert_signal_at(fresh_store, "m_A", datetime.now(timezone.utc) - timedelta(hours=1))
    assert fresh_store.recent_signal_within("m_A", hours=12.0)
    assert not fresh_store.recent_signal_within("m_B", hours=12.0)


def test_record_signal_then_dedupe_blocks_second_call(fresh_store):
    m = _fake_market("m_record")
    fresh_store.record_signal(m, ["narrow_pullback", "breakout"], 2)
    # immediate re-check should now hit the dedupe
    assert fresh_store.recent_signal_within("m_record", hours=12.0)
