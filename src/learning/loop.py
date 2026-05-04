"""Module 7 — daily review scaffold. Computes signal accuracy and asks Claude
for a failure-mode summary; future iterations should feed back into weights."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from src.probability.llm_client import LLMClient
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("learning")

_FAILURE_PROMPT = """\
You review a Polymarket trading agent's last 24h. Given the closed positions
below, identify (a) the top failure modes, (b) which K-line signals seem
unreliable, (c) one or two parameter changes you would try.

Closed positions (CSV: market_id,reason,exit_price,pnl_usdc,signals_at_open):
{closed_block}

Recent skip decisions sample:
{skip_block}

Respond JSON:
{{"failure_modes": ["..."], "weak_signals": ["..."], "suggested_changes": ["..."]}}
"""


class LearningLoop:
    def __init__(self, store: TraceStore, llm: LLMClient) -> None:
        self._store = store
        self._llm = llm

    def daily_review(self) -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=1)
        signal_perf = self._signal_accuracy(since)
        analysis = self._llm_failure_review(since) if self._llm.is_ready() else {}

        log.info(f"Daily review: signals={signal_perf}")
        return {"signal_performance": signal_perf, "llm_analysis": analysis}

    def _signal_accuracy(self, since: datetime) -> dict[str, dict]:
        """Map each signal name to (count, win_rate, avg_pnl) across closed positions."""
        with self._store._conn() as cx:  # noqa: SLF001
            rows = list(cx.execute(
                """SELECT s.signals as signals, p.pnl_usdc as pnl
                   FROM positions p
                   JOIN signals s ON s.token_id = p.token_id
                   WHERE p.closed_at IS NOT NULL AND p.closed_at >= ?""",
                (since.isoformat(),),
            ))
        agg: dict[str, list[float]] = {}
        for r in rows:
            for sig in str(r["signals"] or "").split(","):
                sig = sig.strip()
                if sig:
                    agg.setdefault(sig, []).append(float(r["pnl"] or 0.0))
        out: dict[str, dict] = {}
        for sig, pnls in agg.items():
            wins = sum(1 for p in pnls if p > 0)
            out[sig] = {
                "count": len(pnls),
                "win_rate": wins / len(pnls) if pnls else 0.0,
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            }
        return out

    def _llm_failure_review(self, since: datetime) -> dict:
        with self._store._conn() as cx:  # noqa: SLF001
            closed = list(cx.execute(
                """SELECT p.market_id, p.close_reason, p.exit_price, p.pnl_usdc, s.signals
                   FROM positions p LEFT JOIN signals s ON s.token_id = p.token_id
                   WHERE p.closed_at IS NOT NULL AND p.closed_at >= ?""",
                (since.isoformat(),),
            ))
            skipped = list(cx.execute(
                """SELECT market_id, reason FROM decisions
                   WHERE action='skip' AND ts >= ? ORDER BY id DESC LIMIT 20""",
                (since.isoformat(),),
            ))

        if not closed:
            return {}

        closed_block = "\n".join(
            f"{r['market_id']},{r['close_reason']},{r['exit_price']},{r['pnl_usdc']},{r['signals']}"
            for r in closed
        )
        skip_block = "\n".join(f"{r['market_id']}: {r['reason']}" for r in skipped) or "(none)"

        return self._llm.complete_json(
            _FAILURE_PROMPT.format(closed_block=closed_block, skip_block=skip_block),
            max_tokens=600, temperature=0.3,
        ) or {}

    @staticmethod
    def signal_distribution(store: TraceStore) -> Counter:
        with store._conn() as cx:  # noqa: SLF001
            rows = cx.execute("SELECT signals FROM signals").fetchall()
        c: Counter = Counter()
        for r in rows:
            for sig in str(r["signals"] or "").split(","):
                if sig.strip():
                    c[sig.strip()] += 1
        return c
