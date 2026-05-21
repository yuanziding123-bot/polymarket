"""Collect historical resolved Polymarket markets for ML training.

Two-step process:
  1. Page through Gamma `/markets?closed=true`, parse outcome (which side
     resolved to $1), store metadata.
  2. For each resolved market, download trade history via /trades and persist
     into trades_cache (uses existing infrastructure).

Run with:
    python -m src.data.historical_collector --markets 1000
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Iterator

import httpx

from src.data.polymarket_client import GAMMA_BASE, PolymarketClient
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("hist_collector")

PAGE = 100  # Gamma per-response cap


def _parse_outcome(market: dict) -> str | None:
    """Determine winning side from outcomePrices. Returns 'YES', 'NO', or None."""
    raw_prices = market.get("outcomePrices")
    raw_outcomes = market.get("outcomes")
    if not raw_prices or not raw_outcomes:
        return None
    try:
        prices = [float(p) for p in (json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices)]
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
    except (ValueError, json.JSONDecodeError):
        return None
    if len(prices) != 2 or len(outcomes) != 2:
        return None
    # Polymarket convention: outcomes = ["Yes", "No"], prices = [yes_price, no_price]
    # Resolved markets have prices = [1, 0] or [0, 1] (one side equals $1).
    yes_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() in {"yes", "true"}), None)
    no_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() in {"no", "false"}), None)
    if yes_idx is None or no_idx is None:
        return None
    if prices[yes_idx] >= 0.95:
        return "YES"
    if prices[no_idx] >= 0.95:
        return "NO"
    return None  # Not cleanly resolved (e.g. canceled, in dispute)


def _parse_token_ids(market: dict) -> tuple[str | None, str | None]:
    raw = market.get("clobTokenIds")
    if not raw:
        return None, None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, json.JSONDecodeError):
        return None, None
    outcomes_raw = market.get("outcomes")
    if not outcomes_raw:
        return None, None
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    except (ValueError, json.JSONDecodeError):
        return None, None
    if len(ids) != 2 or len(outcomes) != 2:
        return None, None
    yes_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() in {"yes", "true"}), None)
    no_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() in {"no", "false"}), None)
    if yes_idx is None or no_idx is None:
        return None, None
    return str(ids[yes_idx]), str(ids[no_idx])


def iter_closed_markets(target_count: int, http: httpx.Client) -> Iterator[dict]:
    """Page through Gamma /markets?closed=true until we've yielded target_count
    cleanly-resolved (YES/NO) markets."""
    yielded = 0
    offset = 0
    seen_cids: set[str] = set()
    while yielded < target_count:
        try:
            r = http.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "closed": "true",
                    "limit": PAGE,
                    "offset": offset,
                    "order": "endDate",
                    "ascending": "false",
                },
            )
            r.raise_for_status()
            page = r.json() or []
        except Exception as exc:
            log.warning(f"Gamma fetch failed at offset={offset}: {exc}")
            break
        if not page:
            break
        for m in page:
            cid = m.get("conditionId")
            if not cid or cid in seen_cids:
                continue
            seen_cids.add(cid)
            yielded += 1
            yield m
            if yielded >= target_count:
                break
        if len(page) < PAGE:
            break
        offset += PAGE


def collect_metadata(target_markets: int, store: TraceStore, http: httpx.Client) -> list[dict]:
    """Step 1: scan Gamma and persist metadata. Returns parsed market dicts."""
    log.info(f"Collecting metadata for up to {target_markets} closed markets…")
    collected: list[dict] = []
    skipped_unparsed = 0
    for m in iter_closed_markets(target_markets * 2, http):  # over-fetch to skip dirty rows
        if len(collected) >= target_markets:
            break
        winning_side = _parse_outcome(m)
        if winning_side is None:
            skipped_unparsed += 1
            continue
        yes_token, no_token = _parse_token_ids(m)
        if not yes_token or not no_token:
            skipped_unparsed += 1
            continue
        cid = m["conditionId"]
        store.upsert_market_metadata(
            condition_id=cid,
            question=str(m.get("question") or "")[:200],
            yes_token_id=yes_token,
            no_token_id=no_token,
            winning_side=winning_side,
            closed=True,
            end_date=str(m.get("endDate") or m.get("endDateIso") or ""),
            closed_time=str(m.get("closedTime") or ""),
            last_trade_price=float(m.get("lastTradePrice") or 0.0),
        )
        collected.append({
            "condition_id": cid,
            "yes_token_id": yes_token,
            "no_token_id": no_token,
            "winning_side": winning_side,
            "question": m.get("question"),
        })
    log.info(f"Persisted metadata: {len(collected)} markets ({skipped_unparsed} skipped as unparsed)")
    return collected


def download_trades(markets: list[dict], client: PolymarketClient, store: TraceStore,
                    sleep_between: float = 0.1) -> dict:
    """Step 2: download trades for each market that isn't already cached."""
    log.info(f"Downloading trades for {len(markets)} markets…")
    with store._conn() as cx:  # noqa: SLF001
        cached_cids = {r[0] for r in cx.execute(
            "SELECT DISTINCT condition_id FROM trades_cache"
        ).fetchall()}
    n_new = n_skipped = n_failed = 0
    for i, m in enumerate(markets, 1):
        cid = m["condition_id"]
        if cid in cached_cids:
            n_skipped += 1
            continue
        try:
            trades = client.fetch_market_trades(cid)
        except Exception as exc:
            log.warning(f"[{i}/{len(markets)}] {cid[:14]}: {exc}")
            n_failed += 1
            continue
        if not trades:
            n_failed += 1
            continue
        store.insert_trades(cid, trades)
        n_new += 1
        if i % 20 == 0:
            log.info(f"[{i}/{len(markets)}] new={n_new} cached={n_skipped} failed={n_failed}")
        time.sleep(sleep_between)
    log.info(f"Trade download done: new={n_new} cached={n_skipped} failed={n_failed}")
    return {"new": n_new, "cached": n_skipped, "failed": n_failed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=int, default=1000,
                        help="target number of resolved markets to collect")
    parser.add_argument("--metadata-only", action="store_true",
                        help="skip trade download")
    parser.add_argument("--sleep", type=float, default=0.1,
                        help="seconds between /trades requests (rate-limit safety)")
    args = parser.parse_args()

    store = TraceStore()
    client = PolymarketClient()
    http = httpx.Client(timeout=30.0)

    parsed = collect_metadata(args.markets, store, http)
    if not args.metadata_only:
        download_trades(parsed, client, store, sleep_between=args.sleep)


if __name__ == "__main__":
    main()
