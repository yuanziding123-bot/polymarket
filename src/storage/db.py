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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id TEXT NOT NULL,
    market_id TEXT, entry_price REAL, peak_price REAL,
    size_usdc REAL, opened_at TEXT, expiry TEXT,
    closed_at TEXT, close_reason TEXT, exit_price REAL, pnl_usdc REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(token_id, closed_at);
CREATE TABLE IF NOT EXISTS trades_cache (
    condition_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    side TEXT,
    size REAL NOT NULL,
    price REAL,
    tx_hash TEXT,
    proxy_wallet TEXT,
    PRIMARY KEY (condition_id, asset, timestamp, tx_hash)
);
CREATE INDEX IF NOT EXISTS idx_trades_cond_ts ON trades_cache(condition_id, timestamp);
CREATE TABLE IF NOT EXISTS markets_metadata (
    condition_id TEXT PRIMARY KEY,
    question TEXT,
    yes_token_id TEXT,
    no_token_id TEXT,
    winning_side TEXT,        -- "YES" | "NO" | NULL if unresolved
    closed INTEGER DEFAULT 0,
    end_date TEXT,
    closed_time TEXT,
    last_trade_price REAL,
    last_fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_markets_closed ON markets_metadata(closed, winning_side);
"""


class TraceStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or SETTINGS.sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            self._migrate_positions_schema(cx)
            cx.executescript(_SCHEMA)

    @staticmethod
    def _migrate_positions_schema(cx: sqlite3.Connection) -> None:
        """One-shot migration: old `positions` had `token_id PRIMARY KEY`, which
        blocked re-opening a closed position on the same token. Detect and rebuild
        with an auto-increment id PK while preserving any rows."""
        row = cx.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='positions'"
        ).fetchone()
        if not row:
            return  # fresh DB, normal CREATE will apply
        existing_sql = row["sql"] or ""
        # If existing table already has the new id PK, nothing to do.
        if "id INTEGER PRIMARY KEY AUTOINCREMENT" in existing_sql:
            return
        # Rebuild with new schema; preserve data.
        cx.executescript("""
            ALTER TABLE positions RENAME TO positions_old;
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                market_id TEXT, entry_price REAL, peak_price REAL,
                size_usdc REAL, opened_at TEXT, expiry TEXT,
                closed_at TEXT, close_reason TEXT, exit_price REAL, pnl_usdc REAL
            );
            INSERT INTO positions
                (token_id, market_id, entry_price, peak_price, size_usdc,
                 opened_at, expiry, closed_at, close_reason, exit_price, pnl_usdc)
            SELECT token_id, market_id, entry_price, peak_price, size_usdc,
                   opened_at, expiry, closed_at, close_reason, exit_price, pnl_usdc
            FROM positions_old;
            DROP TABLE positions_old;
            CREATE INDEX IF NOT EXISTS idx_positions_open
                ON positions(token_id, closed_at);
        """)

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

    def recent_signal_within(self, market_id: str, hours: float = 12.0) -> bool:
        """True if this market triggered a (whitelisted) signal within the lookback.
        Used by the live pipeline to avoid re-paying for LLM calls on a market
        that's been in a triggered state for many consecutive scans — mirrors
        the backtest's `dedupe_bars` parameter."""
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._conn() as cx:
            row = cx.execute(
                "SELECT 1 FROM signals WHERE market_id=? AND ts >= ? LIMIT 1",
                (market_id, cutoff),
            ).fetchone()
        return row is not None

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
        """Open a new position OR add to an existing open position on the same token.

        When the same token is bought twice (e.g. a re-trigger after the 12h dedupe),
        we weight-average the entry prices by tokens held and sum the USDC notional.
        This mirrors how a real exchange aggregates fills into a single position —
        Polymarket tokens are fungible. Prior behaviour silently overwrote only
        peak_price and dropped the second buy from position tracking.
        """
        with self._conn() as cx:
            existing = cx.execute(
                "SELECT entry_price, size_usdc, peak_price FROM positions "
                "WHERE token_id=? AND closed_at IS NULL",
                (position.token_id,),
            ).fetchone()
            if existing:
                old_size = float(existing["size_usdc"])
                old_entry = float(existing["entry_price"])
                tokens_old = old_size / old_entry if old_entry > 0 else 0.0
                tokens_new = position.size_usdc / position.entry_price if position.entry_price > 0 else 0.0
                tokens_total = tokens_old + tokens_new
                if tokens_total <= 0:
                    return
                new_size = old_size + position.size_usdc
                new_entry = new_size / tokens_total
                new_peak = max(float(existing["peak_price"]), position.peak_price)
                cx.execute(
                    "UPDATE positions SET entry_price=?, peak_price=?, size_usdc=? "
                    "WHERE token_id=? AND closed_at IS NULL",
                    (new_entry, new_peak, new_size, position.token_id),
                )
            else:
                cx.execute(
                    """INSERT INTO positions(token_id,market_id,entry_price,peak_price,size_usdc,
                       opened_at,expiry) VALUES(?,?,?,?,?,?,?)""",
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

    # ----- trades cache --------------------------------------------------

    def insert_trades(self, condition_id: str, trades: list[dict]) -> int:
        """Bulk-insert trade rows. Duplicates ignored via PRIMARY KEY conflict."""
        if not trades:
            return 0
        rows = [
            (
                condition_id,
                str(t.get("asset") or ""),
                int(t.get("timestamp") or 0),
                str(t.get("side") or ""),
                float(t.get("size") or 0.0),
                float(t.get("price") or 0.0),
                str(t.get("transactionHash") or ""),
                str(t.get("proxyWallet") or ""),
            )
            for t in trades
            if t.get("timestamp")
        ]
        with self._conn() as cx:
            cx.executemany(
                """INSERT OR IGNORE INTO trades_cache
                   (condition_id, asset, timestamp, side, size, price, tx_hash, proxy_wallet)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def upsert_market_metadata(self, condition_id: str, *, question: str,
                                yes_token_id: str, no_token_id: str,
                                winning_side: str | None, closed: bool,
                                end_date: str, closed_time: str | None,
                                last_trade_price: float | None) -> None:
        with self._conn() as cx:
            cx.execute(
                """INSERT INTO markets_metadata
                   (condition_id, question, yes_token_id, no_token_id, winning_side,
                    closed, end_date, closed_time, last_trade_price, last_fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(condition_id) DO UPDATE SET
                     question=excluded.question,
                     yes_token_id=excluded.yes_token_id,
                     no_token_id=excluded.no_token_id,
                     winning_side=excluded.winning_side,
                     closed=excluded.closed,
                     end_date=excluded.end_date,
                     closed_time=excluded.closed_time,
                     last_trade_price=excluded.last_trade_price,
                     last_fetched_at=excluded.last_fetched_at""",
                (condition_id, question, yes_token_id, no_token_id, winning_side,
                 1 if closed else 0, end_date, closed_time, last_trade_price, _now()),
            )

    def get_resolved_markets(self) -> list[sqlite3.Row]:
        """All markets with a known winning_side (resolved)."""
        with self._conn() as cx:
            return list(cx.execute(
                "SELECT * FROM markets_metadata WHERE closed=1 AND winning_side IN ('YES','NO')"
            ))

    def cached_trade_ts_range(self, condition_id: str) -> tuple[int, int] | None:
        """Return (min_ts, max_ts) currently cached for this market, or None."""
        with self._conn() as cx:
            row = cx.execute(
                "SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi FROM trades_cache WHERE condition_id=?",
                (condition_id,),
            ).fetchone()
        if not row or row["lo"] is None:
            return None
        return int(row["lo"]), int(row["hi"])

    def fetch_trades(
        self,
        condition_id: str,
        asset: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> list[sqlite3.Row]:
        sql = "SELECT timestamp, asset, side, size, price FROM trades_cache WHERE condition_id=?"
        args: list = [condition_id]
        if asset:
            sql += " AND asset=?"
            args.append(asset)
        if min_ts is not None:
            sql += " AND timestamp>=?"
            args.append(int(min_ts))
        if max_ts is not None:
            sql += " AND timestamp<=?"
            args.append(int(max_ts))
        sql += " ORDER BY timestamp ASC"
        with self._conn() as cx:
            return list(cx.execute(sql, args))

    # ----- positions / decisions reads ----------------------------------

    def open_positions(self) -> list[sqlite3.Row]:
        with self._conn() as cx:
            return list(cx.execute("SELECT * FROM positions WHERE closed_at IS NULL"))

    def recent_decisions(self, limit: int = 50) -> list[sqlite3.Row]:
        with self._conn() as cx:
            return list(cx.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
