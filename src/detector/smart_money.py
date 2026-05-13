"""Module 2 — SmartMoneyDetector. Five candle/volume signals; trigger when ≥2 fire."""
from __future__ import annotations

from typing import Sequence

from src.data.types import Candle, DetectionResult
from src.utils.logger import get_logger
from src.utils.math_utils import linear_regression, safe_mean

log = get_logger("detector")

MIN_HISTORY = 120  # bars needed for the longest lookback (slow_grind / breakout)

# Whitelist derived from the 300-market backtest (see 回测分析报告 §6.1 update):
#   Only `breakout + narrow_pullback` survives 2x-sample regression-to-mean.
#   `narrow_pullback + vol_spike` flipped to negative sharpe at the same price band.
# We keep vol_spike admissible only as a THIRD confirmation (3-signal score had +1.3 sharpe).
_REQUIRED_CORE = frozenset({"narrow_pullback", "breakout"})


def is_whitelisted_combo(signals: list[str] | set[str]) -> bool:
    """Require both `breakout` AND `narrow_pullback`. Other signals are allowed but
    not required (vol_spike etc. may add as 3rd confirmation)."""
    return _REQUIRED_CORE.issubset(set(signals))


class SmartMoneyDetector:
    """Five behaviour-based price signals. Trigger if ≥ 2 fire.

    Lookbacks follow the design doc:
      slow_grind:        last 120 bars
      vol_trend:         last 60 bars (linear regression)
      narrowing_pullback last 60 vs prior 60 bars
      breakout:          MA60 vs MA120 + last close > MA60*1.03
      vol_spike:         last 5 bars vs prior 60 bars
    """

    def detect(self, candles: Sequence[Candle]) -> DetectionResult:
        if len(candles) < MIN_HISTORY:
            return DetectionResult(triggered=False, score=0, signals=[])

        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        signals: list[str] = []
        if self._slow_grind(closes):
            signals.append("slow_grind")
        if self._volume_trend(volumes):
            signals.append("vol_trend")
        if self._narrowing_pullback(closes):
            signals.append("narrow_pullback")
        if self._breakout(closes):
            signals.append("breakout")
        if self._vol_spike(volumes):
            signals.append("vol_spike")

        score = len(signals)
        return DetectionResult(triggered=score >= 2, score=score, signals=signals)

    # --- signals --------------------------------------------------------

    @staticmethod
    def _slow_grind(closes: Sequence[float]) -> bool:
        window = closes[-120:]
        if window[0] <= 0:
            return False
        total_change = (window[-1] - window[0]) / window[0]
        max_single = max(
            abs(window[i] - window[i - 1]) / window[i - 1]
            for i in range(1, len(window))
            if window[i - 1] > 0
        )
        return total_change > 0.05 and max_single < 0.015

    @staticmethod
    def _volume_trend(volumes: Sequence[float]) -> bool:
        window = volumes[-60:]
        if not any(window):
            return False
        slope, r2 = linear_regression(window)
        return slope > 0 and r2 > 0.5

    @staticmethod
    def _narrowing_pullback(closes: Sequence[float]) -> bool:
        recent = closes[-60:]
        earlier = closes[-120:-60]
        if len(recent) < 60 or len(earlier) < 60:
            return False
        recent_dd = _max_drawdown(recent)
        earlier_dd = _max_drawdown(earlier)
        if earlier_dd == 0:
            return False
        return recent_dd < earlier_dd * 0.6

    @staticmethod
    def _breakout(closes: Sequence[float]) -> bool:
        if len(closes) < 120:
            return False
        ma60 = safe_mean(closes[-60:])
        ma120 = safe_mean(closes[-120:])
        if ma120 == 0:
            return False
        bias = abs(ma60 - ma120) / ma120
        return bias < 0.02 and closes[-1] > ma60 * 1.03

    @staticmethod
    def _vol_spike(volumes: Sequence[float]) -> bool:
        if len(volumes) < 65:
            return False
        recent_avg = safe_mean(volumes[-5:])
        baseline = safe_mean(volumes[-65:-5])
        if baseline == 0:
            return False
        return recent_avg > baseline * 2.5


def _max_drawdown(prices: Sequence[float]) -> float:
    """Largest peak-to-trough drawdown as a positive fraction (0..1)."""
    if not prices:
        return 0.0
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        if peak > 0:
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd
