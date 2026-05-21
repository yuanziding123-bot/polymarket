"""Quick PnL check on all open positions — fetch live prices, compare to entry."""
from __future__ import annotations

import sqlite3
from src.data.polymarket_client import PolymarketClient


def main() -> None:
    cx = sqlite3.connect("data/traces.db")
    cx.row_factory = sqlite3.Row
    rows = list(cx.execute(
        "SELECT market_id, token_id, entry_price, size_usdc, opened_at "
        "FROM positions WHERE closed_at IS NULL ORDER BY opened_at"
    ))
    if not rows:
        print("No open positions.")
        return

    client = PolymarketClient()
    total_entry = 0.0
    total_current = 0.0
    for r in rows:
        candles = client.fetch_price_history(r["token_id"], interval="1m", fidelity=60)
        if not candles:
            print(f"  no candles for {r['token_id'][:10]}")
            continue
        latest = candles[-1].close
        entry = float(r["entry_price"])
        size_usdc = float(r["size_usdc"])
        tokens_held = size_usdc / entry
        current_value = tokens_held * latest
        pnl = current_value - size_usdc
        pnl_pct = (latest - entry) / entry * 100 if entry else 0
        print(f"opened {r['opened_at'][:19]}  entry={entry:.4f}  "
              f"current={latest:.4f}  PnL={pnl_pct:+.2f}%  "
              f"(${size_usdc:.2f} -> ${current_value:.2f})")
        total_entry += size_usdc
        total_current += current_value

    if total_entry:
        agg = (total_current - total_entry) / total_entry * 100
        print(f"\nTotal: ${total_entry:.2f} -> ${total_current:.2f}  ({agg:+.2f}%)")


if __name__ == "__main__":
    main()
