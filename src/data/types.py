"""Domain types shared across modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

Side = Literal["buy", "sell"]
Outcome = Literal["YES", "NO"]


@dataclass
class Market:
    market_id: str
    condition_id: str
    question: str
    description: str
    outcome: Outcome
    token_id: str
    price: float
    volume_24h: float
    liquidity: float
    spread: float
    days_to_expiry: float
    expiry: datetime
    raw: dict = field(default_factory=dict)


@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class DetectionResult:
    triggered: bool
    score: int
    signals: list[str]


@dataclass
class ProbabilityEstimate:
    p_true: float
    components: dict[str, float]
    reasoning: list[str]
    confidence: Literal["low", "medium", "high"]
    uncertainty: str = ""


@dataclass
class TradeDecision:
    market_id: str
    token_id: str
    side: Side
    market_price: float
    p_true: float
    edge: float
    position_size_usdc: float
    action: Literal["buy", "skip"]
    reason: str = ""
    bull_summary: str = ""
    bear_summary: str = ""


@dataclass
class TradeResult:
    executed: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    size: Optional[float] = None
    reason: str = ""


@dataclass
class Position:
    market_id: str
    token_id: str
    entry_price: float
    size_usdc: float
    peak_price: float
    opened_at: datetime
    expiry: datetime

    @property
    def days_to_expiry(self) -> float:
        return (self.expiry - datetime.utcnow()).total_seconds() / 86400.0
