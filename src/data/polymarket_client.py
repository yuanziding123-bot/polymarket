"""Polymarket data + CLOB client wrapper.

Wraps both:
  * Gamma REST API (public market metadata) — used for market discovery
  * py-clob-client (order book, fills, candle history)

The wrapper isolates the rest of the system from SDK details so tests
can substitute a fake. All trading is gated by `dry_run`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx

from config import SETTINGS
from src.data.types import Candle, Market, Outcome
from src.utils.logger import get_logger

log = get_logger("polymarket")

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = SETTINGS.polymarket_host
DATA_API_BASE = "https://data-api.polymarket.com"

# /trades default order is desc-by-timestamp. Docs claim offset max=10000 but the
# production API returns 400 once offset >= 3500, so we cap at 3000 to stay safe.
# At limit=500 that's 7 pages = ~3500 most-recent trades reachable per market.
TRADES_PAGE_SIZE = 500
TRADES_MAX_OFFSET = 3000


@dataclass
class _ClientHandles:
    clob: Any | None
    http: httpx.Client


def _init_clob():
    """Lazy import; SDK only required when keys are configured."""
    if not SETTINGS.polymarket_private_key:
        log.info("CLOB client disabled (no POLYMARKET_PRIVATE_KEY); read-only mode.")
        return None
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        log.warning("py-clob-client not installed; CLOB disabled.")
        return None
    client = ClobClient(
        host=CLOB_BASE,
        chain_id=SETTINGS.polymarket_chain_id,
        key=SETTINGS.polymarket_private_key,
    )
    try:
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
    except Exception as exc:  # pragma: no cover — environment dependent
        log.warning(f"Could not derive CLOB API creds: {exc}")
    return client


class PolymarketClient:
    def __init__(self) -> None:
        self._handles = _ClientHandles(clob=_init_clob(), http=httpx.Client(timeout=20.0))

    # ----- read-only market discovery (Gamma) -----

    def list_active_markets(self, limit: int = 500) -> list[dict]:
        """Pull active, non-archived markets from Gamma."""
        params = {
            "active": "true",
            "archived": "false",
            "closed": "false",
            "limit": limit,
            "order": "volume24hr",
            "ascending": "false",
        }
        try:
            r = self._handles.http.get(f"{GAMMA_BASE}/markets", params=params)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.error(f"Gamma list_active_markets failed: {exc}")
            return []

    def to_markets(self, raw_markets: Iterable[dict]) -> list[Market]:
        """Normalize Gamma payloads into Market objects (one per outcome side).

        Polymarket returns YES/NO token ids inside `clobTokenIds` and
        outcomes/prices as parallel arrays.
        """
        out: list[Market] = []
        for m in raw_markets:
            try:
                outcomes = _parse_json_field(m.get("outcomes"))
                prices = [float(p) for p in _parse_json_field(m.get("outcomePrices") or [])]
                token_ids = _parse_json_field(m.get("clobTokenIds") or [])
                if not outcomes or len(outcomes) != len(prices) or len(prices) != len(token_ids):
                    continue
                expiry = _parse_iso(m.get("endDate") or m.get("end_date_iso"))
                if not expiry:
                    continue
                days_to_expiry = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400.0
                for outcome_label, price, token_id in zip(outcomes, prices, token_ids):
                    side: Outcome = "YES" if str(outcome_label).strip().lower() in {"yes", "true"} else "NO"
                    out.append(
                        Market(
                            market_id=str(m.get("id") or m.get("conditionId")),
                            condition_id=str(m.get("conditionId") or ""),
                            question=str(m.get("question") or ""),
                            description=str(m.get("description") or ""),
                            outcome=side,
                            token_id=str(token_id),
                            price=price,
                            volume_24h=float(m.get("volume24hr") or 0.0),
                            liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0.0),
                            spread=float(m.get("spread") or 0.0),
                            days_to_expiry=days_to_expiry,
                            expiry=expiry,
                            raw=m,
                        )
                    )
            except Exception as exc:
                log.debug(f"Skipping malformed market: {exc}")
        return out

    # ----- price history -----

    def fetch_price_history(self, token_id: str, interval: str = "1h", fidelity: int = 60) -> list[Candle]:
        """Fetch price history from Polymarket's prices-history endpoint.

        Returns synthetic candles where open=close=price (the endpoint provides
        a single price per bucket); volume is filled via fetch_volume_history.
        """
        try:
            r = self._handles.http.get(
                f"{CLOB_BASE}/prices-history",
                params={"market": token_id, "interval": interval, "fidelity": fidelity},
            )
            r.raise_for_status()
            history = r.json().get("history", [])
        except Exception as exc:
            log.warning(f"prices-history failed for {token_id[:10]}…: {exc}")
            return []
        candles: list[Candle] = []
        for point in history:
            ts = datetime.fromtimestamp(int(point["t"]), tz=timezone.utc)
            price = float(point["p"])
            candles.append(Candle(ts=ts, open=price, high=price, low=price, close=price, volume=0.0))
        return candles

    # ----- trades (data-api) -----

    def fetch_market_trades(
        self,
        condition_id: str,
        min_ts: int | None = None,
        max_pages: int = 25,
    ) -> list[dict]:
        """Paginate /trades for a market until min_ts (unix seconds) or offset cap.

        Trades come back desc-by-timestamp; we stop once a page contains a
        trade older than min_ts (the rest are older too).
        """
        if not condition_id:
            return []
        out: list[dict] = []
        offset = 0
        for _ in range(max_pages):
            if offset > TRADES_MAX_OFFSET:
                break
            try:
                r = self._handles.http.get(
                    f"{DATA_API_BASE}/trades",
                    params={
                        "market": condition_id,
                        "limit": TRADES_PAGE_SIZE,
                        "offset": offset,
                        "takerOnly": "false",
                    },
                )
                r.raise_for_status()
                page = r.json() or []
            except Exception as exc:
                log.warning(f"/trades failed for {condition_id[:10]}…: {exc}")
                break
            if not page:
                break
            out.extend(page)
            oldest = page[-1].get("timestamp")
            if min_ts is not None and isinstance(oldest, (int, float)) and oldest < min_ts:
                break
            if len(page) < TRADES_PAGE_SIZE:
                break
            offset += TRADES_PAGE_SIZE
        return out

    # ----- order book / trading -----

    def get_order_book(self, token_id: str) -> dict | None:
        if not self._handles.clob:
            return None
        try:
            return self._handles.clob.get_order_book(token_id)
        except Exception as exc:
            log.warning(f"order book failed for {token_id[:10]}…: {exc}")
            return None

    def post_limit_order(self, token_id: str, price: float, size: float, side: str) -> dict | None:
        """Place a GTC limit order. Caller must check dry-run upstream."""
        if not self._handles.clob:
            raise RuntimeError("CLOB client not configured; cannot trade.")
        from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore

        order_args = OrderArgs(token_id=token_id, price=price, size=size, side=side.upper())
        signed = self._handles.clob.create_order(order_args)
        return self._handles.clob.post_order(signed, OrderType.GTC)

    def cancel_order(self, order_id: str) -> dict | None:
        if not self._handles.clob:
            return None
        return self._handles.clob.cancel(order_id=order_id)

    def list_open_orders(self) -> list[dict]:
        if not self._handles.clob:
            return []
        try:
            return self._handles.clob.get_orders()
        except Exception as exc:
            log.warning(f"list_open_orders failed: {exc}")
            return []


def _parse_json_field(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        import json
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    s = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
