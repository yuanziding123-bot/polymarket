"""Volume reconstruction: bucket /trades into hourly bins aligned to Candle timestamps.

The Polymarket prices-history endpoint returns price-only series. We rebuild
volume by paginating /trades for the market's conditionId and bucket-summing
each trade's `size` (token units) into the candle bucket whose start <= ts < next.

Only trades on the matching `asset` (= token_id of the candle's outcome side)
are counted, otherwise YES-side trades would inflate NO-side volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.data.polymarket_client import PolymarketClient
from src.data.types import Candle
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("volume")


@dataclass
class _Trade:
    timestamp: int
    asset: str
    size: float


def enrich_candles_with_volume(
    candles: list[Candle],
    condition_id: str,
    token_id: str,
    client: PolymarketClient,
    cache: TraceStore | None = None,
) -> list[Candle]:
    """Return new candles with .volume populated.

    Strategy: needed range = [candles[0].ts, candles[-1].ts + bar_seconds]. Use
    cached trades when their range covers the needed window, otherwise pull
    fresh trades from /trades and persist them. Empty/short markets simply get
    zero-volume candles back.
    """
    if not candles:
        return candles

    bar_seconds = _detect_bar_seconds(candles)
    if bar_seconds <= 0:
        return candles

    needed_lo = int(candles[0].ts.timestamp())
    needed_hi = int(candles[-1].ts.timestamp()) + bar_seconds

    trades = _load_trades(condition_id, token_id, needed_lo, needed_hi, client, cache)
    if not trades:
        return candles

    bucket_volumes = _bucket_sum(trades, candles[0].ts.timestamp(), bar_seconds, len(candles))
    enriched = list(candles)
    for i, c in enumerate(enriched):
        if bucket_volumes[i] > 0:
            enriched[i] = Candle(
                ts=c.ts, open=c.open, high=c.high, low=c.low, close=c.close,
                volume=bucket_volumes[i],
            )
    nz = sum(1 for v in bucket_volumes if v > 0)
    log.debug(f"enriched {nz}/{len(candles)} candles with volume for {condition_id[:10]}")
    return enriched


# ----------------------------------------------------------------------

def _detect_bar_seconds(candles: list[Candle]) -> int:
    if len(candles) < 2:
        return 3600
    deltas = [
        int((candles[i + 1].ts - candles[i].ts).total_seconds())
        for i in range(min(10, len(candles) - 1))
    ]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return 3600
    return min(deltas)  # smallest gap = the natural bar size


def _load_trades(
    condition_id: str,
    token_id: str,
    needed_lo: int,
    needed_hi: int,
    client: PolymarketClient,
    cache: TraceStore | None,
) -> list[_Trade]:
    if cache is None:
        raw = client.fetch_market_trades(condition_id, min_ts=needed_lo)
        return _flatten(raw, token_id)

    cached_range = cache.cached_trade_ts_range(condition_id)
    have_full_coverage = cached_range and cached_range[0] <= needed_lo
    if not have_full_coverage:
        raw = client.fetch_market_trades(condition_id, min_ts=needed_lo)
        if raw:
            cache.insert_trades(condition_id, raw)

    rows = cache.fetch_trades(condition_id, asset=token_id, min_ts=needed_lo, max_ts=needed_hi)
    return [_Trade(timestamp=int(r["timestamp"]), asset=r["asset"], size=float(r["size"])) for r in rows]


def _flatten(raw_trades: Iterable[dict], token_id: str) -> list[_Trade]:
    out: list[_Trade] = []
    for t in raw_trades:
        if str(t.get("asset") or "") != token_id:
            continue
        ts = t.get("timestamp")
        size = t.get("size")
        if not ts or size is None:
            continue
        out.append(_Trade(timestamp=int(ts), asset=token_id, size=float(size)))
    return out


def _bucket_sum(
    trades: list[_Trade],
    first_ts_seconds: float,
    bar_seconds: int,
    n_buckets: int,
) -> list[float]:
    buckets = [0.0] * n_buckets
    base = int(first_ts_seconds)
    for t in trades:
        idx = (t.timestamp - base) // bar_seconds
        if 0 <= idx < n_buckets:
            buckets[idx] += t.size
    return buckets
