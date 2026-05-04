"""Module 1 — rule-based pre-filter for the raw Polymarket universe."""
from __future__ import annotations

from dataclasses import dataclass

from src.data.polymarket_client import PolymarketClient
from src.data.types import Market
from src.utils.logger import get_logger

log = get_logger("scanner")


@dataclass(frozen=True)
class FilterConfig:
    min_price: float = 0.05
    max_price: float = 0.50
    min_volume_24h: float = 5_000.0
    min_liquidity: float = 10_000.0
    max_spread_pct: float = 0.05
    min_days_to_expiry: float = 3.0
    max_days_to_expiry: float = 90.0


class MarketScanner:
    def __init__(self, client: PolymarketClient, cfg: FilterConfig | None = None) -> None:
        self._client = client
        self._cfg = cfg or FilterConfig()

    def scan(self, raw_limit: int = 500) -> list[Market]:
        raw = self._client.list_active_markets(limit=raw_limit)
        markets = self._client.to_markets(raw)
        log.info(f"Scanner pulled {len(markets)} outcome rows from {len(raw)} markets")

        passing = [m for m in markets if self._passes(m)]
        deduped = self._dedupe_per_event(passing)
        log.info(f"Scanner {len(passing)} pass filters, {len(deduped)} after YES/NO dedup")
        return deduped

    def _passes(self, m: Market) -> bool:
        c = self._cfg
        if not (c.min_price <= m.price <= c.max_price):
            return False
        if m.volume_24h < c.min_volume_24h:
            return False
        if m.liquidity < c.min_liquidity:
            return False
        spread_pct = (m.spread / m.price) if m.price > 0 else 1.0
        if spread_pct > c.max_spread_pct:
            return False
        if not (c.min_days_to_expiry <= m.days_to_expiry <= c.max_days_to_expiry):
            return False
        return True

    @staticmethod
    def _dedupe_per_event(markets: list[Market]) -> list[Market]:
        """Per condition_id keep the lower-priced side (higher payoff odds)."""
        best: dict[str, Market] = {}
        for m in markets:
            key = m.condition_id or m.market_id
            cur = best.get(key)
            if cur is None or m.price < cur.price:
                best[key] = m
        return list(best.values())
