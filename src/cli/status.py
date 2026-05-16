"""`python main.py status` — at-a-glance snapshot of the running system.

Reads everything from SQLite, no API calls. Safe to run while the main loop is going.
"""
from __future__ import annotations

from datetime import datetime, time, timezone

from config import SETTINGS
from src.storage.db import TraceStore


def show_status() -> None:
    store = TraceStore()
    start_of_day = datetime.combine(datetime.utcnow().date(), time.min, tzinfo=timezone.utc).isoformat()

    with store._conn() as cx:  # noqa: SLF001
        scans = cx.execute("SELECT COUNT(*) AS n, MAX(ts) AS last FROM scans").fetchone()
        scans_today = cx.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(n_signals),0) AS sig, "
            "COALESCE(SUM(n_buys),0) AS buys FROM scans WHERE ts >= ?",
            (start_of_day,),
        ).fetchone()

        sig_count = cx.execute(
            "SELECT COUNT(*) AS n FROM signals WHERE ts >= ?", (start_of_day,)
        ).fetchone()

        decisions = cx.execute(
            """SELECT action, COUNT(*) AS n FROM decisions
               WHERE ts >= ? GROUP BY action""",
            (start_of_day,),
        ).fetchall()
        action_map = {r["action"]: r["n"] for r in decisions}

        edges = cx.execute(
            "SELECT edge FROM decisions WHERE ts >= ? AND action='skip'",
            (start_of_day,),
        ).fetchall()
        edge_vals = sorted(float(r["edge"]) for r in edges)

        open_pos = cx.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(size_usdc),0) AS exp "
            "FROM positions WHERE closed_at IS NULL"
        ).fetchone()

        closed_today = cx.execute(
            """SELECT COUNT(*) AS n, COALESCE(SUM(pnl_usdc),0) AS pnl
               FROM positions WHERE closed_at >= ?""",
            (start_of_day,),
        ).fetchone()

        recent_decisions = cx.execute(
            """SELECT ts, action, market_price, p_true, edge, reason
               FROM decisions ORDER BY id DESC LIMIT 10"""
        ).fetchall()

    print("=" * 70)
    print(f"  Polymarket Agent — Status @ {datetime.utcnow().isoformat()}Z")
    print(f"  Mode: {'LIVE' if SETTINGS.is_live else 'DRY_RUN'} | Bankroll: ${SETTINGS.bankroll_usdc:.0f}")
    print("=" * 70)

    print(f"\nScans:        total={scans['n']:>4}  last={scans['last']}")
    print(f"  today:      {scans_today['n']:>4} cycles, "
          f"{scans_today['sig']:>3} signals triggered, {scans_today['buys']} buys")

    print(f"\nSignals today (whitelisted): {sig_count['n']}")

    print(f"\nDecisions today:")
    print(f"  buy:  {action_map.get('buy', 0)}")
    print(f"  skip: {action_map.get('skip', 0)}")
    if edge_vals:
        median = edge_vals[len(edge_vals) // 2]
        print(f"  skip-edge range: {edge_vals[0]:+.4f} … {edge_vals[-1]:+.4f}  median={median:+.4f}")

    print(f"\nPositions:")
    print(f"  open:   {open_pos['n']} (exposure ${open_pos['exp']:.2f})")
    print(f"  closed today: {closed_today['n']} (realized PnL ${closed_today['pnl']:+.2f})")

    if recent_decisions:
        print(f"\nLast 10 decisions:")
        print(f"  {'time':<20} {'action':<6} {'mkt':>6} {'p_true':>7} {'edge':>8}  reason")
        for r in recent_decisions:
            ts = r['ts'][:19].replace('T', ' ')
            edge = float(r['edge'] or 0)
            print(f"  {ts:<20} {r['action']:<6} {float(r['market_price']):>6.3f} "
                  f"{float(r['p_true']):>7.3f} {edge:>+8.4f}  {(r['reason'] or '')[:40]}")
    print()
