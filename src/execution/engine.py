"""Module 5 — order execution. Honours dry_run by recording-only."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from config import SETTINGS
from src.data.polymarket_client import PolymarketClient
from src.data.types import Market, Position, TradeDecision, TradeResult
from src.notify.telegram import Notifier
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("execution")

SLIPPAGE_TOLERANCE = 0.005  # 0.5% — design doc default


class ExecutionEngine:
    def __init__(
        self,
        client: PolymarketClient,
        store: TraceStore,
        notifier: Notifier | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._notifier = notifier or Notifier()

    def execute(self, decision: TradeDecision, market: Market) -> TradeResult:
        if decision.action != "buy" or decision.position_size_usdc <= 0:
            return TradeResult(executed=False, reason=decision.reason or "no_action")

        limit_price = self._limit_price(decision.market_price, decision.side)
        size_tokens = round(decision.position_size_usdc / max(limit_price, 1e-6), 2)

        mode = "live" if SETTINGS.is_live else "dry_run"

        if not SETTINGS.is_live:
            order_id = f"dry-{uuid4().hex[:12]}"
            log.info(
                f"[dry_run] BUY {market.question[:60]} | px={limit_price:.4f} "
                f"size={size_tokens} usdc={decision.position_size_usdc:.2f}"
            )
            self._store.record_order(
                market_id=decision.market_id, token_id=decision.token_id, side=decision.side,
                price=limit_price, size=size_tokens, mode=mode,
                order_id=order_id, status="dry_run", raw=None,
            )
            self._open_position(decision, market, limit_price)
            self._notifier.position_opened(
                question=market.question, side=decision.side, price=limit_price,
                size_usdc=decision.position_size_usdc, edge=decision.edge, mode=mode,
            )
            return TradeResult(executed=True, order_id=order_id, filled_price=limit_price,
                               size=size_tokens, reason="dry_run")

        try:
            resp = self._client.post_limit_order(
                token_id=decision.token_id, price=limit_price, size=size_tokens, side=decision.side,
            ) or {}
        except Exception as exc:
            log.error(f"Live order failed: {exc}")
            self._store.record_order(
                market_id=decision.market_id, token_id=decision.token_id, side=decision.side,
                price=limit_price, size=size_tokens, mode=mode, order_id=None,
                status=f"error:{exc}", raw=None,
            )
            return TradeResult(executed=False, reason=str(exc))

        order_id = str(resp.get("orderID") or resp.get("id") or "")
        self._store.record_order(
            market_id=decision.market_id, token_id=decision.token_id, side=decision.side,
            price=limit_price, size=size_tokens, mode=mode, order_id=order_id,
            status="placed", raw=resp,
        )
        self._open_position(decision, market, limit_price)
        self._notifier.position_opened(
            question=market.question, side=decision.side, price=limit_price,
            size_usdc=decision.position_size_usdc, edge=decision.edge, mode=mode,
        )
        return TradeResult(executed=True, order_id=order_id, filled_price=limit_price,
                           size=size_tokens, reason="placed")

    def _open_position(self, decision: TradeDecision, market: Market, fill_price: float) -> None:
        pos = Position(
            market_id=decision.market_id,
            token_id=decision.token_id,
            entry_price=fill_price,
            size_usdc=decision.position_size_usdc,
            peak_price=fill_price,
            opened_at=datetime.now(timezone.utc),
            expiry=market.expiry,
        )
        self._store.upsert_position(pos)

    @staticmethod
    def _limit_price(market_price: float, side: str) -> float:
        if side == "buy":
            return round(min(0.99, market_price * (1.0 + SLIPPAGE_TOLERANCE)), 4)
        return round(max(0.01, market_price * (1.0 - SLIPPAGE_TOLERANCE)), 4)
