"""Monitor loop (every 30s by default): risk-manage open positions."""
from __future__ import annotations

from src.data.polymarket_client import PolymarketClient
from src.risk.manager import RiskManager
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("monitor")


def run_monitor_once(client: PolymarketClient, store: TraceStore, risk: RiskManager) -> None:
    actions = risk.evaluate()
    if not actions:
        return
    log.info(f"Risk monitor → {len(actions)} close actions")
    for action in actions:
        size_tokens = _position_size_tokens(store, action.token_id, action.current_price)
        risk.close(action, size_tokens)


def _position_size_tokens(store: TraceStore, token_id: str, current_price: float) -> float:
    """Approx tokens held = entry_size_usdc / entry_price."""
    with store._conn() as cx:  # noqa: SLF001
        row = cx.execute(
            "SELECT entry_price, size_usdc FROM positions WHERE token_id=?",
            (token_id,),
        ).fetchone()
    if not row:
        return 0.0
    entry_price = float(row["entry_price"]) or current_price
    return round(float(row["size_usdc"]) / max(entry_price, 1e-6), 2)
