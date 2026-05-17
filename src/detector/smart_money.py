"""Module 2 — SmartMoneyDetector. Five candle/volume signals; trigger when ≥2 fire."""
from __future__ import annotations

from typing import Sequence

from src.data.types import Candle, DetectionResult
from src.utils.logger import get_logger
from src.utils.math_utils import linear_regression, safe_mean

log = get_logger("detector")

MIN_HISTORY = 120  # bars needed for the longest lookback (slow_grind / breakout)

# Whitelist re-derived 2026-05-17 after detector recalibration backtest:
#   With relaxed thresholds, the alpha shifted from 0.10-0.20 to <0.10.
#   Both `breakout+narrow_pullback` (sharpe +0.139) and `narrow_pullback+vol_spike`
#   (sharpe +0.104) showed positive alpha at n=99-140 in the <0.10 band.
#   Every positive-sharpe bucket contained narrow_pullback as the anchor signal.
#   `breakout+vol_spike` alone (no narrow_pullback) was negative at sharpe -0.451.
_ANCHOR = "narrow_pullback"
_CONFIRMS = frozenset({"breakout", "vol_spike"})


def is_whitelisted_combo(signals: list[str] | set[str]) -> bool:
    """Require `narrow_pullback` AND at least one of `breakout` or `vol_spike`."""
    sigset = set(signals)
    return _ANCHOR in sigset and bool(sigset & _CONFIRMS)


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

    # Thresholds re-calibrated 2026-05-17 after live diagnostic showed 4/5 signals
    # never fired on real Polymarket data (slow probability markets, not stock-like
    # momentum). Goal: get trigger rate from <0.1/day to ~5-10/day while preserving
    # the directional alpha pattern (narrow_pullback as anchor signal).

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
        # was: total>5% AND single<1.5%
        return total_change > 0.03 and max_single < 0.020

    @staticmethod
    def _volume_trend(volumes: Sequence[float]) -> bool:
        window = volumes[-60:]
        if not any(window):
            return False
        slope, r2 = linear_regression(window)
        # was: R²>0.5 (too strict for lumpy Polymarket volume)
        return slope > 0 and r2 > 0.3

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
        # unchanged — this is the only signal that fires on live data
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
        # was: bias<2% AND close>MA60*1.03 (3% breakout)
        return bias < 0.03 and closes[-1] > ma60 * 1.015

    @staticmethod
    def _vol_spike(volumes: Sequence[float]) -> bool:
        if len(volumes) < 65:
            return False
        recent_avg = safe_mean(volumes[-5:])
        baseline = safe_mean(volumes[-65:-5])
        if baseline == 0:
            return False
        # was: 2.5x baseline (too rare on Polymarket)
        return recent_avg > baseline * 1.5


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
