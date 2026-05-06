"""Sliding-window backtest of SmartMoneyDetector against historical Polymarket prices.

For each market we replay candles[120:len-horizon], call detector.detect(candles[:i]),
and when triggered we log (price[i] -> price[i+horizon]) as one trial. Consecutive
triggers within `dedupe_bars` collapse to a single trial to avoid double counting
neighbouring windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.data.polymarket_client import PolymarketClient
from src.data.types import Candle, Market
from src.data.volume import enrich_candles_with_volume
from src.detector.smart_money import MIN_HISTORY, SmartMoneyDetector
from src.scanner.market_scanner import FilterConfig, MarketScanner
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("backtest")


@dataclass
class Trial:
    market_id: str
    outcome: str
    question: str
    bar_index: int
    entry_price: float
    exit_price: float
    horizon_bars: int
    signals: tuple[str, ...]
    score: int

    @property
    def return_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class MarketUniverse:
    """A backtest pool of markets with historical candles already fetched."""
    markets: list[Market]
    candle_map: dict[str, list[Candle]]  # token_id -> candles


class BacktestEngine:
    def __init__(
        self,
        client: PolymarketClient,
        detector: SmartMoneyDetector | None = None,
        scanner: MarketScanner | None = None,
        cache: TraceStore | None = None,
        enrich_volume: bool = True,
    ) -> None:
        self._client = client
        self._detector = detector or SmartMoneyDetector()
        self._cache = cache
        self._enrich_volume = enrich_volume
        # Loosen filters for backtest — we want any liquid, non-degenerate market.
        self._scanner = scanner or MarketScanner(
            client,
            FilterConfig(
                min_price=0.02, max_price=0.98,
                min_volume_24h=1_000.0, min_liquidity=2_000.0,
                max_spread_pct=0.20,
                min_days_to_expiry=0.0, max_days_to_expiry=365.0,
            ),
        )

    # ------------------------------------------------------------------

    def build_universe(
        self,
        n_markets: int = 50,
        raw_limit: int = 500,
        history_interval: str = "max",
        fidelity: int = 60,
    ) -> MarketUniverse:
        """Pull a batch of markets and fetch historical candles for each."""
        markets = self._scanner.scan(raw_limit=raw_limit)
        log.info(f"Backtest pool candidates: {len(markets)}")

        universe_markets: list[Market] = []
        candle_map: dict[str, list[Candle]] = {}
        for market in markets:
            if len(universe_markets) >= n_markets:
                break
            candles = self._client.fetch_price_history(
                market.token_id, interval=history_interval, fidelity=fidelity,
            )
            if len(candles) < MIN_HISTORY + 24:  # need history + at least 1 day forward
                continue
            if self._enrich_volume:
                candles = enrich_candles_with_volume(
                    candles, condition_id=market.condition_id,
                    token_id=market.token_id, client=self._client, cache=self._cache,
                )
            universe_markets.append(market)
            candle_map[market.token_id] = candles
        log.info(f"Universe assembled: {len(universe_markets)} markets, "
                 f"avg {sum(len(v) for v in candle_map.values()) // max(len(candle_map), 1)} candles")
        return MarketUniverse(markets=universe_markets, candle_map=candle_map)

    def run(
        self,
        universe: MarketUniverse,
        horizon_bars: int = 24,
        dedupe_bars: int = 12,
    ) -> list[Trial]:
        """Replay each market and emit trials.

        horizon_bars: forward window over which we measure return
        dedupe_bars: minimum spacing between trials on the same market
        """
        trials: list[Trial] = []
        for market in universe.markets:
            candles = universe.candle_map[market.token_id]
            trials.extend(self._replay_market(market, candles, horizon_bars, dedupe_bars))
        log.info(f"Total trials: {len(trials)}")
        return trials

    def _replay_market(
        self,
        market: Market,
        candles: list[Candle],
        horizon_bars: int,
        dedupe_bars: int,
    ) -> list[Trial]:
        out: list[Trial] = []
        last_trial_idx = -dedupe_bars  # so the first trigger always counts
        for i in range(MIN_HISTORY, len(candles) - horizon_bars):
            window = candles[:i]
            result = self._detector.detect(window)
            if not result.triggered:
                continue
            if i - last_trial_idx < dedupe_bars:
                continue
            entry = candles[i].close
            exit_ = candles[i + horizon_bars].close
            if entry <= 0 or exit_ <= 0:
                continue
            out.append(Trial(
                market_id=market.market_id,
                outcome=market.outcome,
                question=market.question[:80],
                bar_index=i,
                entry_price=entry,
                exit_price=exit_,
                horizon_bars=horizon_bars,
                signals=tuple(sorted(result.signals)),
                score=result.score,
            ))
            last_trial_idx = i
        return out


def iter_signal_keys(trials: Iterable[Trial]) -> set[tuple[str, ...]]:
    """Distinct signal-combination keys present in the trials."""
    return {t.signals for t in trials}
