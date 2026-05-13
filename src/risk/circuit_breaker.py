"""Pre-trade safety gate. Called by the pipeline before every buy decision.

Four independent rules:
  1. Daily loss limit — cumulative realised PnL for today (UTC) must be > -X% bankroll
  2. Total open exposure cap — sum of open position size_usdc must be < Y% bankroll
  3. Concurrent position cap — count of open positions must be < N
  4. Consecutive-loss cooldown — last K closed positions all losses → pause for T seconds

A trade is permitted only when ALL four checks pass. Each `check()` call returns
a structured verdict so the pipeline can log/notify on the specific block.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone

from config import SETTINGS
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("circuit_breaker")


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""

    @classmethod
    def ok(cls) -> "Verdict":
        return cls(True)

    @classmethod
    def block(cls, reason: str) -> "Verdict":
        return cls(False, reason)


class CircuitBreaker:
    def __init__(self, store: TraceStore, bankroll_usdc: float | None = None) -> None:
        self._store = store
        self._bankroll = bankroll_usdc or SETTINGS.bankroll_usdc

    # ----- public API ---------------------------------------------------

    def check(self, prospective_size_usdc: float = 0.0) -> Verdict:
        for rule in (
            self._check_daily_loss,
            self._check_total_exposure,
            self._check_concurrent_positions,
            self._check_consecutive_losses,
        ):
            verdict = rule(prospective_size_usdc)
            if not verdict.allowed:
                log.warning(f"Circuit breaker BLOCK: {verdict.reason}")
                return verdict
        return Verdict.ok()

    # ----- rules --------------------------------------------------------

    def _check_daily_loss(self, _: float) -> Verdict:
        loss = self._today_realised_pnl()
        limit = -self._bankroll * SETTINGS.max_daily_loss_pct
        if loss <= limit:
            return Verdict.block(
                f"daily loss {loss:+.2f} USDC ≤ limit {limit:+.2f} "
                f"({SETTINGS.max_daily_loss_pct:.0%} of bankroll)"
            )
        return Verdict.ok()

    def _check_total_exposure(self, prospective_size_usdc: float) -> Verdict:
        open_exposure = self._open_exposure_usdc()
        projected = open_exposure + prospective_size_usdc
        cap = self._bankroll * SETTINGS.max_total_exposure_pct
        if projected > cap:
            return Verdict.block(
                f"open exposure {open_exposure:.2f} + new {prospective_size_usdc:.2f} "
                f"= {projected:.2f} > cap {cap:.2f}"
            )
        return Verdict.ok()

    def _check_concurrent_positions(self, _: float) -> Verdict:
        n = len(self._store.open_positions())
        if n >= SETTINGS.max_concurrent_positions:
            return Verdict.block(
                f"{n} open positions ≥ cap {SETTINGS.max_concurrent_positions}"
            )
        return Verdict.ok()

    def _check_consecutive_losses(self, _: float) -> Verdict:
        recent = self._recent_closed_pnls(limit=SETTINGS.max_consecutive_losses)
        if len(recent) < SETTINGS.max_consecutive_losses:
            return Verdict.ok()
        if not all(pnl < 0 for _, pnl in recent):
            return Verdict.ok()
        # All recent N were losses. Check cooldown.
        latest_closed_at = recent[0][0]
        seconds_since = (datetime.now(timezone.utc) - latest_closed_at).total_seconds()
        if seconds_since < SETTINGS.consecutive_loss_cooldown_seconds:
            remaining = SETTINGS.consecutive_loss_cooldown_seconds - int(seconds_since)
            return Verdict.block(
                f"last {SETTINGS.max_consecutive_losses} trades all lost; "
                f"cooldown {remaining}s remaining"
            )
        return Verdict.ok()

    # ----- DB helpers ---------------------------------------------------

    def _today_realised_pnl(self) -> float:
        start = datetime.combine(datetime.utcnow().date(), time.min, tzinfo=timezone.utc)
        with self._store._conn() as cx:  # noqa: SLF001
            row = cx.execute(
                """SELECT COALESCE(SUM(pnl_usdc), 0) AS pnl FROM positions
                   WHERE closed_at IS NOT NULL AND closed_at >= ?""",
                (start.isoformat(),),
            ).fetchone()
        return float(row["pnl"] or 0.0)

    def _open_exposure_usdc(self) -> float:
        with self._store._conn() as cx:  # noqa: SLF001
            row = cx.execute(
                "SELECT COALESCE(SUM(size_usdc), 0) AS total FROM positions WHERE closed_at IS NULL"
            ).fetchone()
        return float(row["total"] or 0.0)

    def _recent_closed_pnls(self, limit: int) -> list[tuple[datetime, float]]:
        with self._store._conn() as cx:  # noqa: SLF001
            rows = cx.execute(
                """SELECT closed_at, pnl_usdc FROM positions
                   WHERE closed_at IS NOT NULL
                   ORDER BY closed_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        out: list[tuple[datetime, float]] = []
        for r in rows:
            try:
                dt = datetime.fromisoformat(r["closed_at"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                out.append((dt, float(r["pnl_usdc"] or 0.0)))
            except (ValueError, TypeError):
                continue
        return out
