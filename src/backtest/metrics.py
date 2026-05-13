"""Aggregate backtest trials into per-signal performance metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Iterable

from src.backtest.engine import Trial


@dataclass
class SignalStats:
    label: str
    n: int
    hit_rate: float
    avg_return: float
    median_return: float
    std_return: float
    sharpe_per_trial: float          # mean / stdev — not annualised
    avg_entry_price: float

    def as_row(self) -> list:
        return [
            self.label, self.n,
            f"{self.hit_rate:.2%}",
            f"{self.avg_return:+.4f}",
            f"{self.median_return:+.4f}",
            f"{self.std_return:.4f}",
            f"{self.sharpe_per_trial:+.3f}",
            f"{self.avg_entry_price:.3f}",
        ]


@dataclass
class BacktestReport:
    overall: SignalStats
    by_signal: list[SignalStats] = field(default_factory=list)         # any single signal present
    by_combo: list[SignalStats] = field(default_factory=list)          # exact signal combo
    by_score: list[SignalStats] = field(default_factory=list)          # score = signal count
    by_price_band: list[SignalStats] = field(default_factory=list)     # entry price bucket
    by_band_x_combo: list[SignalStats] = field(default_factory=list)   # (band, combo) interaction

    @property
    def header(self) -> list[str]:
        return ["label", "n", "hit_rate", "avg_ret", "median_ret",
                "std_ret", "sharpe_trial", "avg_entry_px"]


# Price bands from the design doc — strategy A targets 0.07-0.19 specifically.
PRICE_BANDS = [
    ("p<0.10", 0.0, 0.10),
    ("p<0.10-0.20", 0.10, 0.20),  # the alleged sweet spot
    ("p<0.20-0.30", 0.20, 0.30),
    ("p<0.30-0.50", 0.30, 0.50),
    ("p>=0.50", 0.50, 1.01),
]


def _band_for(price: float) -> str:
    for label, lo, hi in PRICE_BANDS:
        if lo <= price < hi:
            return label
    return "unknown"


def aggregate(trials: list[Trial]) -> BacktestReport:
    if not trials:
        return BacktestReport(overall=_stats("overall", []))

    overall = _stats("overall", trials)

    # by individual signal (a trial can contribute to multiple buckets)
    seen_signals: dict[str, list[Trial]] = {}
    for t in trials:
        for s in t.signals:
            seen_signals.setdefault(s, []).append(t)
    by_signal = [_stats(s, ts) for s, ts in sorted(seen_signals.items())]

    # by exact combo
    combos: dict[tuple[str, ...], list[Trial]] = {}
    for t in trials:
        combos.setdefault(t.signals, []).append(t)
    by_combo = sorted(
        (_stats("+".join(k), ts) for k, ts in combos.items()),
        key=lambda s: s.n, reverse=True,
    )[:10]  # top 10 combos by frequency

    # by signal-count score
    by_score_buckets: dict[int, list[Trial]] = {}
    for t in trials:
        by_score_buckets.setdefault(t.score, []).append(t)
    by_score = [_stats(f"score={k}", ts) for k, ts in sorted(by_score_buckets.items())]

    # by entry-price band
    by_band: dict[str, list[Trial]] = {}
    for t in trials:
        by_band.setdefault(_band_for(t.entry_price), []).append(t)
    by_price_band = [_stats(b, by_band[b]) for b, _, _ in PRICE_BANDS if b in by_band]

    # by (band, combo) — only show buckets with n>=5 to filter noise
    by_band_combo: dict[str, list[Trial]] = {}
    for t in trials:
        key = f"{_band_for(t.entry_price)} | {'+'.join(t.signals)}"
        by_band_combo.setdefault(key, []).append(t)
    by_band_x_combo = sorted(
        (_stats(k, ts) for k, ts in by_band_combo.items() if len(ts) >= 5),
        key=lambda s: s.sharpe_per_trial, reverse=True,
    )

    return BacktestReport(
        overall=overall, by_signal=by_signal,
        by_combo=by_combo, by_score=by_score,
        by_price_band=by_price_band, by_band_x_combo=by_band_x_combo,
    )


def _stats(label: str, trials: Iterable[Trial]) -> SignalStats:
    rs = [t.return_pct for t in trials]
    if not rs:
        return SignalStats(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rs_sorted = sorted(rs)
    median = rs_sorted[len(rs_sorted) // 2]
    avg = mean(rs)
    sd = pstdev(rs) if len(rs) > 1 else 0.0
    sharpe = avg / sd if sd > 0 else 0.0
    hit = sum(1 for r in rs if r > 0) / len(rs)
    avg_px = mean(t.entry_price for t in trials)
    return SignalStats(
        label=label, n=len(rs), hit_rate=hit,
        avg_return=avg, median_return=median, std_return=sd,
        sharpe_per_trial=sharpe, avg_entry_price=avg_px,
    )


def render_table(rows: list[list]) -> str:
    if not rows:
        return "(no rows)"
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "  "
    return "\n".join(sep.join(str(r[i]).ljust(widths[i]) for i in range(len(r))) for r in rows)
