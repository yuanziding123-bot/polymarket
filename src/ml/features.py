"""Feature engineering for the ML probability estimator.

For each (market, time_index) tuple we extract a fixed-length feature vector
from the candle + trade history *up to* that time. No look-ahead.

Reconstructs hourly bars directly from the trades_cache so we don't need
separate candle storage and the bar prices are guaranteed consistent with
the trade flow (last trade in the hour = bar close).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

BAR_SECONDS = 3600  # hourly bars

# Minimum bars of history before we'll emit a sample (longest lookback is 120)
MIN_HISTORY_BARS = 120
# Forward window for the label (24 hours)
LABEL_HORIZON_BARS = 24
# Step between sample points within one market (avoid near-duplicate samples)
SAMPLE_STRIDE_BARS = 6


@dataclass
class HourlyBar:
    ts: int            # bucket start unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float       # total token size traded in bar
    buy_volume: float   # BUY-side size
    sell_volume: float  # SELL-side size
    n_trades: int


def reconstruct_bars(trades: list[dict], asset: str | None = None) -> list[HourlyBar]:
    """Bucket trades into hourly OHLC bars. Trades are expected sorted ASC."""
    if not trades:
        return []
    bars: dict[int, dict] = {}
    for t in trades:
        if asset is not None and t.get("asset") != asset:
            continue
        ts = int(t["timestamp"])
        bucket = (ts // BAR_SECONDS) * BAR_SECONDS
        price = float(t["price"])
        size = float(t["size"])
        side = str(t.get("side") or "").upper()
        b = bars.get(bucket)
        if b is None:
            b = bars[bucket] = {
                "ts": bucket, "open": price, "high": price, "low": price, "close": price,
                "volume": 0.0, "buy_volume": 0.0, "sell_volume": 0.0, "n_trades": 0,
            }
        b["high"] = max(b["high"], price)
        b["low"] = min(b["low"], price)
        b["close"] = price  # later trades overwrite
        b["volume"] += size
        if side == "BUY":
            b["buy_volume"] += size
        elif side == "SELL":
            b["sell_volume"] += size
        b["n_trades"] += 1

    # Fill missing bars (forward-fill close, zero volume) so windows are contiguous
    if not bars:
        return []
    ts_sorted = sorted(bars.keys())
    first, last = ts_sorted[0], ts_sorted[-1]
    out: list[HourlyBar] = []
    prev_close = bars[first]["open"]
    for ts in range(first, last + BAR_SECONDS, BAR_SECONDS):
        b = bars.get(ts)
        if b is None:
            out.append(HourlyBar(ts=ts, open=prev_close, high=prev_close, low=prev_close,
                                 close=prev_close, volume=0.0, buy_volume=0.0,
                                 sell_volume=0.0, n_trades=0))
        else:
            out.append(HourlyBar(**b))
            prev_close = b["close"]
    return out


# ----------------------------------------------------------------------

FEATURE_NAMES = [
    # price levels
    "price",
    "log_price",
    # returns over windows
    "ret_1h", "ret_6h", "ret_24h", "ret_72h", "ret_120h",
    # moving averages
    "ma_5", "ma_20", "ma_60", "ma_120",
    # MA ratios (catch crossover dynamics)
    "ma5_over_ma60", "ma20_over_ma60", "ma60_over_ma120",
    # volatility (std of returns)
    "vol_5", "vol_20", "vol_60",
    # drawdown
    "dd_60", "dd_120",
    # momentum normalized by vol
    "momentum_z_20", "momentum_z_60",
    # acceleration
    "accel",
    # volume
    "vol_total_5", "vol_total_20", "vol_total_60",
    "vol_ratio_5_60",
    "vol_zscore_60",
    # buy/sell imbalance
    "buy_ratio_20", "buy_ratio_60",
    "buy_sell_imbalance_5",
    # activity
    "n_trades_5", "n_trades_60",
    # bar shape
    "high_low_range_5",
]

SHORT_HORIZON_FEATURE_NAMES = FEATURE_NAMES + ["days_to_resolution", "log_days_to_resolution"]


def extract_features(bars: list[HourlyBar], i: int) -> np.ndarray | None:
    """Return a feature vector at bar index `i`, or None if not enough history."""
    if i < MIN_HISTORY_BARS:
        return None

    closes = np.array([b.close for b in bars[: i + 1]], dtype=float)
    volumes = np.array([b.volume for b in bars[: i + 1]], dtype=float)
    buy_vols = np.array([b.buy_volume for b in bars[: i + 1]], dtype=float)
    sell_vols = np.array([b.sell_volume for b in bars[: i + 1]], dtype=float)
    n_trades_arr = np.array([b.n_trades for b in bars[: i + 1]], dtype=float)
    highs = np.array([b.high for b in bars[: i + 1]], dtype=float)
    lows = np.array([b.low for b in bars[: i + 1]], dtype=float)

    price = closes[-1]
    if price <= 0 or price >= 1:
        return None  # degenerate

    # ---- returns ----
    def ret(n):
        if i < n or closes[-n - 1] <= 0:
            return 0.0
        return float(np.log(price / closes[-n - 1]))

    ret_1, ret_6, ret_24, ret_72, ret_120 = (ret(n) for n in (1, 6, 24, 72, 120))

    # ---- moving averages ----
    def ma(n):
        return float(closes[-n:].mean()) if i >= n else float(closes.mean())

    ma_5, ma_20, ma_60, ma_120 = (ma(n) for n in (5, 20, 60, 120))

    # ---- vol ----
    def vol(n):
        if i < n + 1:
            return 0.0
        r = np.diff(np.log(np.clip(closes[-n - 1:], 1e-6, None)))
        return float(r.std())

    vol_5, vol_20, vol_60 = (vol(n) for n in (5, 20, 60))

    # ---- drawdown ----
    def dd(n):
        window = closes[-n:]
        peak = window.max()
        return float((peak - window[-1]) / peak) if peak > 0 else 0.0

    dd_60, dd_120 = (dd(n) for n in (60, 120))

    # ---- momentum z-score ----
    mom_z_20 = ret_24 / vol_20 if vol_20 > 1e-6 else 0.0
    mom_z_60 = ret_72 / vol_60 if vol_60 > 1e-6 else 0.0

    # ---- acceleration ----
    accel = ret_6 - ret(12) if i >= 12 else 0.0

    # ---- volume features ----
    def vol_total(n):
        return float(volumes[-n:].sum()) if i >= n else float(volumes.sum())

    vol_total_5, vol_total_20, vol_total_60 = (vol_total(n) for n in (5, 20, 60))
    vol_ratio_5_60 = vol_total_5 / (vol_total_60 + 1e-6)
    vol_baseline_mean = volumes[-60:-5].mean() if i >= 60 else volumes.mean()
    vol_baseline_std = volumes[-60:-5].std() if i >= 60 else 1.0
    vol_z = (volumes[-5:].mean() - vol_baseline_mean) / (vol_baseline_std + 1e-6)

    # ---- buy/sell imbalance ----
    def buy_ratio(n):
        b = buy_vols[-n:].sum()
        s = sell_vols[-n:].sum()
        total = b + s
        return float(b / total) if total > 0 else 0.5

    buy_ratio_20 = buy_ratio(20)
    buy_ratio_60 = buy_ratio(60)
    imb_5 = (buy_vols[-5:].sum() - sell_vols[-5:].sum()) / (volumes[-5:].sum() + 1e-6)

    # ---- activity ----
    n_trades_5 = float(n_trades_arr[-5:].sum())
    n_trades_60 = float(n_trades_arr[-60:].sum())

    # ---- bar shape ----
    high_low_range_5 = float(((highs[-5:] - lows[-5:]) / np.clip(closes[-5:], 1e-6, None)).mean())

    return np.array([
        price, np.log(price),
        ret_1, ret_6, ret_24, ret_72, ret_120,
        ma_5, ma_20, ma_60, ma_120,
        ma_5 / (ma_60 + 1e-6), ma_20 / (ma_60 + 1e-6), ma_60 / (ma_120 + 1e-6),
        vol_5, vol_20, vol_60,
        dd_60, dd_120,
        mom_z_20, mom_z_60,
        accel,
        vol_total_5, vol_total_20, vol_total_60,
        vol_ratio_5_60, float(vol_z),
        buy_ratio_20, buy_ratio_60, float(imb_5),
        n_trades_5, n_trades_60,
        high_low_range_5,
    ], dtype=float)


# ----------------------------------------------------------------------
# label = price moves up at least UP_THRESH in horizon

UP_THRESH = 0.01   # 1% — filters small noise moves
HORIZON_BARS = LABEL_HORIZON_BARS  # 24h


def make_label(bars: list[HourlyBar], i: int) -> int | None:
    """Return 1 if bars[i+24].close > bars[i].close * (1+UP_THRESH), else 0."""
    if i + HORIZON_BARS >= len(bars):
        return None
    entry = bars[i].close
    exit_ = bars[i + HORIZON_BARS].close
    if entry <= 0:
        return None
    return 1 if exit_ > entry * (1 + UP_THRESH) else 0


def make_forward_return(bars: list[HourlyBar], i: int) -> float | None:
    """Real 24h forward return (entry → exit)."""
    if i + HORIZON_BARS >= len(bars):
        return None
    entry = bars[i].close
    exit_ = bars[i + HORIZON_BARS].close
    if entry <= 0:
        return None
    return (exit_ - entry) / entry


# ----------------------------------------------------------------------

def load_trades_from_cache(db_path: str, condition_id: str) -> list[dict]:
    """Read all trades for a market sorted by timestamp."""
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT timestamp, asset, side, size, price FROM trades_cache "
        "WHERE condition_id=? ORDER BY timestamp ASC",
        (condition_id,),
    ).fetchall()
    cx.close()
    return [dict(r) for r in rows]


def list_cached_markets(db_path: str, min_trades: int = 200) -> list[str]:
    """Return condition_ids with at least min_trades cached."""
    cx = sqlite3.connect(db_path)
    rows = cx.execute(
        "SELECT condition_id FROM trades_cache GROUP BY condition_id "
        "HAVING COUNT(*) >= ?",
        (min_trades,),
    ).fetchall()
    cx.close()
    return [r[0] for r in rows]


def load_resolved_markets(db_path: str, min_trades: int = 200) -> list[dict]:
    """Return resolved markets that also have trade data cached."""
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        """SELECT m.condition_id, m.yes_token_id, m.no_token_id, m.winning_side, m.question
           FROM markets_metadata m
           WHERE m.closed=1 AND m.winning_side IN ('YES','NO')
             AND m.condition_id IN (
               SELECT condition_id FROM trades_cache
               GROUP BY condition_id HAVING COUNT(*) >= ?
             )""",
        (min_trades,),
    ).fetchall()
    cx.close()
    return [dict(r) for r in rows]


def resolution_label_for_token(token_id: str, market: dict) -> int | None:
    """Return 1 if this token resolved to $1, 0 if it resolved to $0, None if unknown."""
    winning = market.get("winning_side")
    if winning == "YES":
        if token_id == market["yes_token_id"]:
            return 1
        if token_id == market["no_token_id"]:
            return 0
    elif winning == "NO":
        if token_id == market["no_token_id"]:
            return 1
        if token_id == market["yes_token_id"]:
            return 0
    return None


def extract_short_horizon_features(bars: list[HourlyBar], i: int,
                                     days_to_resolution: float) -> np.ndarray | None:
    """Like extract_features but appends days_to_resolution + log version."""
    base = extract_features(bars, i)
    if base is None:
        return None
    return np.concatenate([
        base,
        np.array([days_to_resolution,
                  float(np.log(max(0.01, days_to_resolution)))], dtype=float),
    ])


SHORT_HORIZON_BARS = 6     # predict 6h ahead
SHORT_UP_THRESH = 0.01     # > 1% up to count as positive label
LATE_STAGE_DAYS_MAX = 14   # only sample from markets with ≤ 14 days to resolution


def make_short_horizon_label(bars: list[HourlyBar], i: int) -> int | None:
    if i + SHORT_HORIZON_BARS >= len(bars):
        return None
    entry = bars[i].close
    exit_ = bars[i + SHORT_HORIZON_BARS].close
    if entry <= 0:
        return None
    return 1 if exit_ > entry * (1 + SHORT_UP_THRESH) else 0


def make_short_horizon_return(bars: list[HourlyBar], i: int) -> float | None:
    """Real 6h forward log return."""
    if i + SHORT_HORIZON_BARS >= len(bars):
        return None
    entry = bars[i].close
    exit_ = bars[i + SHORT_HORIZON_BARS].close
    if entry <= 0:
        return None
    return (exit_ - entry) / entry


def iter_short_horizon_dataset(db_path: str, min_trades: int = 200,
                                 late_stage_days_max: float = LATE_STAGE_DAYS_MAX):
    """Emit samples ONLY from late-stage markets (≤ N days to resolution).

    Yields (features, label, forward_return, market_id, token_id, bar_ts).
    """
    from datetime import datetime, timezone
    markets = load_resolved_markets(db_path, min_trades)
    cutoff_seconds = late_stage_days_max * 86400.0

    for m in markets:
        # Need to know when resolution happened to compute days_to_resolution
        cx = sqlite3.connect(db_path)
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT closed_time, end_date FROM markets_metadata WHERE condition_id=?",
            (m["condition_id"],),
        ).fetchone()
        cx.close()
        if not row:
            continue
        resolution_ts = _parse_resolution_ts(row["closed_time"], row["end_date"])
        if resolution_ts is None:
            continue

        trades = load_trades_from_cache(db_path, m["condition_id"])
        if not trades:
            continue
        for asset in (m["yes_token_id"], m["no_token_id"]):
            bars = reconstruct_bars(trades, asset=asset)
            if len(bars) < MIN_HISTORY_BARS + SHORT_HORIZON_BARS:
                continue
            label = resolution_label_for_token(asset, m)
            if label is None:
                continue
            for i in range(MIN_HISTORY_BARS, len(bars) - SHORT_HORIZON_BARS, SAMPLE_STRIDE_BARS):
                seconds_left = resolution_ts - bars[i].ts
                if seconds_left <= 0 or seconds_left > cutoff_seconds:
                    continue
                days_to_resolution = seconds_left / 86400.0
                feats = extract_short_horizon_features(bars, i, days_to_resolution)
                if feats is None:
                    continue
                short_label = make_short_horizon_label(bars, i)
                short_ret = make_short_horizon_return(bars, i)
                if short_label is None or short_ret is None:
                    continue
                yield feats, short_label, short_ret, m["condition_id"], asset, bars[i].ts


def _parse_resolution_ts(closed_time: str | None, end_date: str | None) -> int | None:
    """Parse ISO timestamp to unix seconds; prefer closed_time over end_date.

    Polymarket returns multiple formats; normalise:
      '2026-03-19 23:20:15+00'   → '2026-03-19T23:20:15+00:00'
      '2026-03-19T23:20:15Z'     → '2026-03-19T23:20:15+00:00'
      '2026-03-19T23:20:15+00:00' (already fine)
    """
    from datetime import datetime
    for s in (closed_time, end_date):
        if not s:
            continue
        cleaned = s.strip().replace(" ", "T").replace("Z", "+00:00")
        # Short tz like '+00' must become '+00:00'
        if len(cleaned) >= 3 and cleaned[-3] in {"+", "-"} and ":" not in cleaned[-3:]:
            cleaned = cleaned + ":00"
        try:
            return int(datetime.fromisoformat(cleaned).timestamp())
        except (ValueError, TypeError):
            continue
    return None


def iter_resolution_dataset(db_path: str, min_trades: int = 200):
    """Yield (features, label, market_id, token_id, bar_ts) using REAL resolution outcomes.

    Unlike iter_dataset (which uses 24h forward return as label), here the label
    is whether this specific token eventually resolved to $1.
    """
    markets = load_resolved_markets(db_path, min_trades)
    for m in markets:
        trades = load_trades_from_cache(db_path, m["condition_id"])
        if not trades:
            continue
        for asset in (m["yes_token_id"], m["no_token_id"]):
            bars = reconstruct_bars(trades, asset=asset)
            if len(bars) < MIN_HISTORY_BARS + 1:
                continue
            label = resolution_label_for_token(asset, m)
            if label is None:
                continue
            # Sample throughout the market's life (not just final bar).
            # Skip the last HORIZON_BARS so we have a forward-window distance
            # from settlement (avoid resolution-day data leak).
            for i in range(MIN_HISTORY_BARS, len(bars) - HORIZON_BARS, SAMPLE_STRIDE_BARS):
                feats = extract_features(bars, i)
                if feats is None:
                    continue
                yield feats, label, m["condition_id"], asset, bars[i].ts


def iter_dataset(db_path: str, min_trades: int = 200):
    """Yield (features, label, forward_return, market_id, bar_ts) across all cached markets."""
    market_ids = list_cached_markets(db_path, min_trades)
    for cid in market_ids:
        trades = load_trades_from_cache(db_path, cid)
        if not trades:
            continue
        assets = {t["asset"] for t in trades if t["asset"]}
        for asset in assets:
            bars = reconstruct_bars(trades, asset=asset)
            if len(bars) < MIN_HISTORY_BARS + HORIZON_BARS:
                continue
            for i in range(MIN_HISTORY_BARS, len(bars) - HORIZON_BARS, SAMPLE_STRIDE_BARS):
                feats = extract_features(bars, i)
                if feats is None:
                    continue
                label = make_label(bars, i)
                if label is None:
                    continue
                fwd = make_forward_return(bars, i)
                if fwd is None:
                    continue
                yield feats, label, fwd, cid, bars[i].ts
