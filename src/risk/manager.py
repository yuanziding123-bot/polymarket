"""Module 6 — position risk monitor (stop-loss, trailing, take-profit, expiry)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from config import SETTINGS
from src.data.polymarket_client import PolymarketClient
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("risk")


@dataclass(frozen=True)
class RiskRules:
    stop_loss_pct: float = 0.40
    take_profit_pct: float = 0.60
    trailing_stop_pct: float = 0.15
    trailing_arm_pnl: float = 0.20  # arm trailing once unrealised PnL >= 20%
    force_close_days: float = 1.0


@dataclass
class CloseAction:
    token_id: str
    market_id: str
    reason: str
    current_price: float
    pnl_pct: float


class RiskManager:
    def __init__(
        self,
        client: PolymarketClient,
        store: TraceStore,
        rules: RiskRules | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._rules = rules or RiskRules()

    def evaluate(self) -> list[CloseAction]:
        actions: list[CloseAction] = []
        rows = self._store.open_positions()
        if not rows:
            return actions

        for row in rows:
            current_price = self._latest_price(row["token_id"])
            if current_price is None:
                continue

            entry = float(row["entry_price"])
            peak = max(float(row["peak_price"]), current_price)
            if peak > float(row["peak_price"]):
                # update high-water mark for future trailing-stop checks
                self._store_peak(row["token_id"], peak)

            pnl_pct = (current_price - entry) / entry if entry > 0 else 0.0
            days_left = _days_left(row["expiry"])

            reason = self._classify(pnl_pct, peak, current_price, days_left)
            if reason:
                actions.append(CloseAction(
                    token_id=row["token_id"], market_id=row["market_id"],
                    reason=reason, current_price=current_price, pnl_pct=pnl_pct,
                ))
        return actions

    def close(self, action: CloseAction, size_tokens: float) -> None:
        """Issue (or fake) a sell order and persist the close."""
        mode = "live" if SETTINGS.is_live else "dry_run"
        order_id = None
        status = "dry_run"
        if SETTINGS.is_live:
            try:
                resp = self._client.post_limit_order(
                    token_id=action.token_id,
                    price=max(0.01, round(action.current_price * 0.995, 4)),
                    size=size_tokens, side="sell",
                ) or {}
                order_id = str(resp.get("orderID") or resp.get("id") or "")
                status = "placed"
            except Exception as exc:
                log.error(f"Close failed for {action.token_id[:10]}: {exc}")
                status = f"error:{exc}"

        self._store.record_order(
            market_id=action.market_id, token_id=action.token_id, side="sell",
            price=action.current_price, size=size_tokens, mode=mode,
            order_id=order_id, status=status, raw=None,
        )
        # crude PnL: pct * size_usdc (approx, ignores partial fills)
        pnl_usdc = round(action.pnl_pct * _row_size(self._store, action.token_id), 2)
        self._store.close_position(action.token_id, action.reason,
                                   action.current_price, pnl_usdc)
        log.info(f"Closed {action.token_id[:10]} reason={action.reason} pnl={action.pnl_pct:+.2%}")

    # ------------------------------------------------------------------

    def _classify(
        self, pnl_pct: float, peak_price: float, current_price: float, days_left: float
    ) -> str | None:
        r = self._rules
        if days_left <= r.force_close_days:
            return "near_expiry"
        if pnl_pct <= -r.stop_loss_pct:
            return "stop_loss"
        if pnl_pct >= r.take_profit_pct:
            return "take_profit"
        if pnl_pct >= r.trailing_arm_pnl and peak_price > 0:
            drawdown = (peak_price - current_price) / peak_price
            if drawdown >= r.trailing_stop_pct:
                return "trailing_stop"
        return None

    def _latest_price(self, token_id: str) -> float | None:
        book = self._client.get_order_book(token_id)
        if not book:
            return None
        try:
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            if bids and asks:
                return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0
            if asks:
                return float(asks[0]["price"])
            if bids:
                return float(bids[0]["price"])
        except (KeyError, ValueError, TypeError):
            return None
        return None

    def _store_peak(self, token_id: str, peak: float) -> None:
        # Inline update — keeps risk-monitor stateless across runs.
        with self._store._conn() as cx:  # noqa: SLF001 — internal helper
            cx.execute("UPDATE positions SET peak_price=? WHERE token_id=? AND closed_at IS NULL",
                       (peak, token_id))


def _days_left(expiry_iso: str) -> float:
    try:
        dt = datetime.fromisoformat(expiry_iso)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - datetime.now(timezone.utc)).total_seconds() / 86400.0


def _row_size(store: TraceStore, token_id: str) -> float:
    with store._conn() as cx:  # noqa: SLF001
        row = cx.execute("SELECT size_usdc FROM positions WHERE token_id=?", (token_id,)).fetchone()
        return float(row["size_usdc"]) if row else 0.0


def all_open_token_ids(store: TraceStore) -> Iterable[str]:
    return (row["token_id"] for row in store.open_positions())
