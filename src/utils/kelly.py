"""Kelly criterion for binary prediction markets."""
from __future__ import annotations


def kelly_fraction(p_true: float, p_market: float) -> float:
    """Full Kelly fraction. Returns 0 if no edge."""
    if p_true <= p_market or p_market <= 0 or p_market >= 1:
        return 0.0
    b = (1.0 - p_market) / p_market
    q = 1.0 - p_true
    f = (b * p_true - q) / b
    return max(0.0, f)


def kelly_position_usdc(
    p_true: float,
    p_market: float,
    bankroll: float,
    fraction_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    """Conservative 1/4 Kelly with hard cap on per-trade fraction."""
    full = kelly_fraction(p_true, p_market)
    capped = min(full * fraction_multiplier, max_fraction)
    return round(bankroll * capped, 2)
