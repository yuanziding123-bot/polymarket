"""End-to-end backtest entry: build universe, run engine, aggregate, export CSV."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from config import ROOT
from src.backtest.engine import BacktestEngine, Trial
from src.backtest.metrics import BacktestReport, aggregate, render_table
from src.data.polymarket_client import PolymarketClient
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("backtest_runner")


def run_backtest(
    n_markets: int = 30,
    horizon_bars: int = 24,
    dedupe_bars: int = 12,
    raw_limit: int = 500,
    history_interval: str = "max",
    fidelity: int = 60,
    csv_out: Path | None = None,
) -> tuple[list[Trial], BacktestReport]:
    client = PolymarketClient()
    cache = TraceStore()  # reuse the main SQLite for trades_cache too
    engine = BacktestEngine(client, cache=cache, enrich_volume=True)

    universe = engine.build_universe(
        n_markets=n_markets, raw_limit=raw_limit,
        history_interval=history_interval, fidelity=fidelity,
    )
    trials = engine.run(universe, horizon_bars=horizon_bars, dedupe_bars=dedupe_bars)
    report = aggregate(trials)

    if csv_out is None:
        csv_out = ROOT / "data" / f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    _export_trials(trials, csv_out)
    log.info(f"Wrote {len(trials)} trials → {csv_out}")
    return trials, report


def print_report(report: BacktestReport, horizon_bars: int) -> None:
    print(f"\n=== Backtest report (horizon = {horizon_bars} bars) ===")
    print(f"\nOverall: {report.overall.as_row()}")

    print("\nBy individual signal (a trial may appear in multiple rows):")
    rows = [report.header] + [s.as_row() for s in report.by_signal]
    print(render_table(rows))

    print("\nTop signal combinations:")
    rows = [report.header] + [s.as_row() for s in report.by_combo]
    print(render_table(rows))

    print("\nBy score (number of signals firing):")
    rows = [report.header] + [s.as_row() for s in report.by_score]
    print(render_table(rows))


def _export_trials(trials: list[Trial], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["market_id", "outcome", "question", "bar_index",
                    "entry_price", "exit_price", "return_pct",
                    "horizon_bars", "score", "signals"])
        for t in trials:
            w.writerow([t.market_id, t.outcome, t.question, t.bar_index,
                        f"{t.entry_price:.4f}", f"{t.exit_price:.4f}",
                        f"{t.return_pct:+.4f}", t.horizon_bars, t.score,
                        "+".join(t.signals)])
