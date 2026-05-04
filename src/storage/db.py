"""SQLite-backed trace store. One DB file, multiple tables."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config import SETTINGS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    n_raw INTEGER, n_filtered INTEGER, n_signals INTEGER, n_buys INTEGER
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    market_id TEXT, token_id TEXT, question TEXT,
    price REAL, signals TEXT, score INTEGER
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    market_id TEXT, token_id TEXT, action TEXT,
    market_price REAL, p_true REAL, edge REAL,
    position_size_usdc REAL, reason TEXT,
    bull_summary TEXT, bear_summary TEXT,
    components_json TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    market_id TEXT, token_id TEXT, side TEXT,
    price REAL, size REAL, mode TEXT,
    order_id TEXT, status TEXT, raw_json TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    token_id TEXT PRIMARY KEY,
    market_id TEXT, entry_price REAL, peak_price REAL,
    size_usdc REAL, opened_at TEXT, expiry TEXT,
    closed_at TEXT, close_reason TEXT, exit_price REAL, pnl_usdc REAL
);
"""


class TraceStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or SETTINGS.sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            cx.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        cx = sqlite3.connect(self._path)
        cx.row_factory = sqlite3.Row
        try:
            yield cx
            cx.commit()
        finally:
            cx.close()

    # ----- writes -------------------------------------------------------

    def record_scan(self, n_raw: int, n_filtered: int, n_signals: int, n_buys: int) -> None:
        with self._conn() as cx:
            cx.execute(
                "INSERT INTO scans(ts, n_raw, n_filtered, n_signals, n_buys) VALUES(?,?,?,?,?)",
                (_now(), n_raw, n_filtered, n_signals, n_buys),
            )

    def record_signal(self, market: Any, signals: list[str], score: int) -> None:
        with self._conn() as cx:
            cx.execute(
                "INSERT INTO signals(ts,market_id,token_id,question,price,signals,score) VALUES(?,?,?,?,?,?,?)",
                (_now(), market.market_id, market.token_id, market.question, market.price,
                 ",".join(signals), score),
            )

    def record_decision(self, decision: Any, components: dict[str, float]) -> None:
        with self._conn() as cx:
            cx.execute(
                """INSERT INTO decisions(ts,market_id,token_id,action,market_price,p_true,edge,
                   position_size_usdc,reason,bull_summary,bear_summary,components_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (_now(), decision.market_id, decision.token_id, decision.action,
                 decision.market_price, decision.p_true, decision.edge,
                 decision.position_size_usdc, decision.reason,
                 decision.bull_summary, decision.bear_summary,
                 json.dumps(components)),
            )

    def record_order(self, market_id: str, token_id: str, side: str, price: float,
                     size: float, mode: str, order_id: str | None, status: str,
                     raw: dict | None) -> None:
        with self._conn() as cx:
            cx.execute(
                """INSERT INTO orders(ts,market_id,token_id,side,price,size,mode,order_id,status,raw_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (_now(), market_id, token_id, side, price, size, mode,
                 order_id, status, json.dumps(raw or {})),
            )

    def upsert_position(self, position: Any) -> None:
        with self._conn() as cx:
            cx.execute(
                """INSERT INTO positions(token_id,market_id,entry_price,peak_price,size_usdc,
                   opened_at,expiry) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(token_id) DO UPDATE SET peak_price=excluded.peak_price""",
                (position.token_id, position.market_id, position.entry_price,
                 position.peak_price, position.size_usdc,
                 position.opened_at.isoformat(), position.expiry.isoformat()),
            )

    def close_position(self, token_id: str, reason: str, exit_price: float, pnl_usdc: float) -> None:
        with self._conn() as cx:
            cx.execute(
                """UPDATE positions SET closed_at=?, close_reason=?, exit_price=?, pnl_usdc=?
                   WHERE token_id=? AND closed_at IS NULL""",
                (_now(), reason, exit_price, pnl_usdc, token_id),
            )

    # ----- reads --------------------------------------------------------

    def open_positions(self) -> list[sqlite3.Row]:
        with self._conn() as cx:
            return list(cx.execute("SELECT * FROM positions WHERE closed_at IS NULL"))

    def recent_decisions(self, limit: int = 50) -> list[sqlite3.Row]:
        with self._conn() as cx:
            return list(cx.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
