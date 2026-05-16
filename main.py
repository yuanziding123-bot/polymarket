"""CLI entry point.

Usage:
  python main.py scan-once              # one main-loop cycle
  python main.py monitor-once            # one risk-monitor cycle
  python main.py review                  # daily learning review
  python main.py run                     # start scheduler (main + monitor)
  python main.py run --live              # same, with live trading enabled
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from config import SETTINGS
from src.backtest.runner import print_alpha_summary, print_report, run_backtest
from src.cli.status import show_status
from src.data.polymarket_client import PolymarketClient
from src.learning.loop import LearningLoop
from src.pipeline.main_pipeline import build_pipeline, run_once
from src.pipeline.monitor_pipeline import run_monitor_once
from src.probability.llm_client import LLMClient
from src.risk.manager import RiskManager
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("main")


def _apply_live_flag(args: argparse.Namespace) -> None:
    if getattr(args, "live", False):
        os.environ["RUN_MODE"] = "live"
        # Reload settings module-level constant
        import importlib
        import config as cfg
        importlib.reload(cfg)


def cmd_scan_once(_: argparse.Namespace) -> None:
    components = build_pipeline()
    run_once(components)


def cmd_monitor_once(_: argparse.Namespace) -> None:
    from src.notify.telegram import Notifier
    client = PolymarketClient()
    store = TraceStore()
    notifier = Notifier()
    risk = RiskManager(client, store, notifier=notifier)
    run_monitor_once(client, store, risk)


def cmd_review(_: argparse.Namespace) -> None:
    store = TraceStore()
    llm = LLMClient()
    loop = LearningLoop(store, llm)
    result = loop.daily_review()
    log.info(f"Review → {result}")


def cmd_status(_: argparse.Namespace) -> None:
    show_status()


def cmd_backtest(args: argparse.Namespace) -> None:
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    results = run_backtest(
        n_markets=args.markets,
        horizons=horizons,
        dedupe_bars=args.dedupe,
    )
    for h in horizons:
        trials, report = results[h]
        print_report(report, horizon_bars=h)
    print_alpha_summary(results)


def cmd_run(args: argparse.Namespace) -> None:
    """Run main + monitor loops on a scheduler until interrupted."""
    from apscheduler.schedulers.background import BackgroundScheduler

    components = build_pipeline()
    risk = RiskManager(components.client, components.store, notifier=components.notifier)

    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        lambda: run_once(components),
        "interval", seconds=SETTINGS.main_loop_interval,
        next_run_time=_now(), id="main_loop", max_instances=1,
    )
    sched.add_job(
        lambda: run_monitor_once(components.client, components.store, risk),
        "interval", seconds=SETTINGS.monitor_loop_interval,
        next_run_time=_now(), id="monitor_loop", max_instances=1,
    )
    sched.start()
    log.info(f"Scheduler started | mode={'live' if SETTINGS.is_live else 'dry_run'} "
             f"main={SETTINGS.main_loop_interval}s monitor={SETTINGS.monitor_loop_interval}s")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("Shutting down…")
    finally:
        sched.shutdown(wait=False)


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polymarket-agent")
    parser.add_argument("--live", action="store_true", help="enable live trading (overrides RUN_MODE=dry_run)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan-once", help="run a single main-loop iteration")
    sub.add_parser("monitor-once", help="run a single risk-monitor iteration")
    sub.add_parser("review", help="run the daily learning review")
    sub.add_parser("run", help="start the scheduler (main + monitor)")
    sub.add_parser("status", help="print SQLite snapshot of today's activity")

    bt = sub.add_parser("backtest", help="replay SmartMoneyDetector on historical candles")
    bt.add_argument("--markets", type=int, default=30, help="number of markets to sample")
    bt.add_argument("--horizons", type=str, default="24",
                    help="comma-separated forward bars to measure return (e.g. 6,24,168)")
    bt.add_argument("--dedupe", type=int, default=12, help="min bars between trials per market")

    args = parser.parse_args(argv)
    _apply_live_flag(args)

    {
        "scan-once": cmd_scan_once,
        "monitor-once": cmd_monitor_once,
        "review": cmd_review,
        "run": cmd_run,
        "backtest": cmd_backtest,
        "status": cmd_status,
    }[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
