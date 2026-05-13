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
    horizons: list[int] | None = None,
    dedupe_bars: int = 12,
    raw_limit: int = 1500,
    history_interval: str = "max",
    fidelity: int = 60,
    csv_out: Path | None = None,
) -> dict[int, tuple[list[Trial], BacktestReport]]:
    """Build universe once, replay engine for each horizon, return per-horizon results."""
    horizons = horizons or [24]
    client = PolymarketClient()
    cache = TraceStore()  # reuse the main SQLite for trades_cache too
    engine = BacktestEngine(client, cache=cache, enrich_volume=True)

    universe = engine.build_universe(
        n_markets=n_markets, raw_limit=raw_limit,
        history_interval=history_interval, fidelity=fidelity,
    )

    out: dict[int, tuple[list[Trial], BacktestReport]] = {}
    ts_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for h in horizons:
        trials = engine.run(universe, horizon_bars=h, dedupe_bars=dedupe_bars)
        report = aggregate(trials)
        out[h] = (trials, report)

        path = csv_out or (ROOT / "data" / f"backtest_{ts_label}_h{h}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        _export_trials(trials, path)
        log.info(f"Horizon {h}: {len(trials)} trials → {path}")
    return out


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

    print("\nBy entry price band:")
    rows = [report.header] + [s.as_row() for s in report.by_price_band]
    print(render_table(rows))

    print("\nTop (band, combo) buckets by sharpe (n>=5):")
    rows = [report.header] + [s.as_row() for s in report.by_band_x_combo[:15]]
    print(render_table(rows))


def print_alpha_summary(results: dict[int, tuple[list[Trial], BacktestReport]]) -> None:
    """Cross-horizon summary: any (signal-combo, horizon, band) with positive sharpe & n>=15?"""
    print("\n" + "=" * 70)
    print("ALPHA HUNT — positive-sharpe buckets with n>=15")
    print("=" * 70)
    hits = []
    for h, (_, report) in sorted(results.items()):
        for s in report.by_band_x_combo:
            if s.n >= 15 and s.sharpe_per_trial > 0:
                hits.append((h, s))
    if not hits:
        print("(none — no statistically-meaningful positive sharpe bucket found)")
        return
    print(f"{'horizon':<9}{'label':<55}{'n':>4}{'hit%':>8}{'avg_ret':>10}{'sharpe':>9}")
    for h, s in sorted(hits, key=lambda x: x[1].sharpe_per_trial, reverse=True):
        print(f"{h:<9}{s.label[:54]:<55}{s.n:>4}{s.hit_rate:>8.1%}"
              f"{s.avg_return:>+10.4f}{s.sharpe_per_trial:>+9.3f}")


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
