"""Short-horizon ML estimator. Predicts 6h-forward return using a LightGBM
regressor trained on late-stage markets (≤ 14 days to resolution).

Best backtest config from sweep:
  pred > 0.20 → sharpe +0.198 per trade, avg +18.7%

Semantics:
- model output is a *return* (e.g. 0.20 = "expect +20% price move in 6h"),
  NOT a probability.
- We reinterpret p_true = market_price + predicted_return so the existing
  edge = p_true - market_price math falls out to: edge = predicted_return.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from config import ROOT
from src.data.polymarket_client import PolymarketClient
from src.data.types import DetectionResult, Market, ProbabilityEstimate
from src.ml.features import (
    LATE_STAGE_DAYS_MAX,
    MIN_HISTORY_BARS,
    SHORT_HORIZON_FEATURE_NAMES,
    extract_short_horizon_features,
    reconstruct_bars,
)
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("ml_sh_estimator")

DEFAULT_MODEL_PATH = ROOT / "data" / "ml_model_short_horizon_regressor.txt"
SELL_MODEL_PATH = ROOT / "data" / "ml_model_short_horizon_sell_classifier.txt"

# Decision threshold. Sweep showed both 0.10 (n=643, sharpe +0.118) and 0.20
# (n=202, sharpe +0.198) give similar *annualised* sharpe (~11). Using 0.10
# for the dry-run because pred>0.20 was rare enough that live triggers
# became scarce — lower threshold keeps the system actively trading so we
# can gather real PnL data faster.
DECISION_THRESHOLD_PRED_RETURN = 0.10

# Sell threshold: P(drop >2% in 6h) > this → close position.
# Backtest: P(drop)>0.5 fires on 5.2% of held bars, 55.8% accurate,
# saves ~6.7% average forward loss vs holding.
SELL_THRESHOLD_DROP_PROB = 0.50


class MLShortHorizonEstimator:
    def __init__(
        self,
        client: PolymarketClient,
        store: TraceStore | None = None,
        model_path: Path | None = None,
        sell_model_path: Path | None = None,
    ) -> None:
        path = model_path or DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Short-horizon buy model not found at {path}. "
                "Run `python -m src.ml.train_short_horizon` first."
            )
        self._model = lgb.Booster(model_file=str(path))
        sell_path = sell_model_path or SELL_MODEL_PATH
        self._sell_model = lgb.Booster(model_file=str(sell_path)) if sell_path.exists() else None
        if self._sell_model is None:
            log.warning(f"Sell model not found at {sell_path}; exit signal disabled.")
        else:
            log.info(f"Loaded sell classifier from {sell_path}")
        self._client = client
        self._store = store or TraceStore()
        log.info(f"Loaded short-horizon regressor from {path}")

    def estimate(
        self,
        market: Market,
        detection: DetectionResult,
        correlated_price: float | None = None,
    ) -> ProbabilityEstimate:
        # Gate by market age: model only trained on late-stage data
        if market.days_to_expiry > LATE_STAGE_DAYS_MAX:
            return self._skip(
                market,
                reason=f"days_to_expiry={market.days_to_expiry:.1f} > {LATE_STAGE_DAYS_MAX}",
            )
        if market.days_to_expiry <= 0.25:  # 6h
            return self._skip(market, reason="market too close to expiry for 6h horizon")

        bars = self._build_bars(market)
        if bars is None or len(bars) < MIN_HISTORY_BARS:
            return self._skip(market, reason="insufficient_history")

        i = len(bars) - 1
        features = extract_short_horizon_features(bars, i, market.days_to_expiry)
        if features is None:
            return self._skip(market, reason="feature_error")

        predicted_return = float(self._model.predict(features.reshape(1, -1))[0])

        # Confidence is graded so we can later analyse by tier.
        # The pipeline buys whenever predicted_return >= DECISION_THRESHOLD.
        if predicted_return >= 0.20:
            confidence = "high"
        elif predicted_return >= DECISION_THRESHOLD_PRED_RETURN:
            confidence = "medium"
        else:
            confidence = "low"

        p_true = market.price + predicted_return  # edge falls out to predicted_return
        return ProbabilityEstimate(
            p_true=max(0.0, min(1.0, p_true)),
            components={"predicted_return": predicted_return, "market_price": market.price},
            reasoning=[
                f"Short-horizon regressor on {len(bars)} bars + days_to_resolution={market.days_to_expiry:.1f}",
                f"Predicted 6h return: {predicted_return:+.4f}",
                f"Decision rule: trade when predicted_return > {DECISION_THRESHOLD_PRED_RETURN}",
            ],
            confidence=confidence,
            uncertainty="6h_forward_late_stage",
        )

    def _skip(self, market: Market, reason: str) -> ProbabilityEstimate:
        return ProbabilityEstimate(
            p_true=market.price,
            components={"market_price": market.price, "skipped": 1.0},
            reasoning=[f"Skipped: {reason}"],
            confidence="low",
            uncertainty=reason,
        )

    def estimate_sell(self, token_id: str, condition_id: str,
                       days_to_resolution: float) -> dict | None:
        """Predict P(price drops >2% in next 6h) for a held position.

        Returns dict with `drop_probability`, `should_sell`, and `reason`.
        Returns None if model unavailable or features can't be built.
        """
        if self._sell_model is None:
            return None
        bars = self._build_bars_by_ids(condition_id, token_id)
        if bars is None or len(bars) < MIN_HISTORY_BARS:
            return None
        i = len(bars) - 1
        features = extract_short_horizon_features(bars, i, days_to_resolution)
        if features is None:
            return None
        drop_prob = float(self._sell_model.predict(features.reshape(1, -1))[0])
        should_sell = drop_prob >= SELL_THRESHOLD_DROP_PROB
        return {
            "drop_probability": drop_prob,
            "should_sell": should_sell,
            "reason": (f"ml_sell: P(drop>2%)={drop_prob:.3f} "
                       f">= {SELL_THRESHOLD_DROP_PROB}" if should_sell else
                       f"ml_sell: P(drop)={drop_prob:.3f} below threshold"),
        }

    def _build_bars_by_ids(self, condition_id: str, token_id: str):
        """Same as _build_bars but takes IDs directly (for held positions)."""
        cached = self._store.fetch_trades(condition_id, asset=token_id)
        trades = [dict(r) for r in cached]
        if not trades:
            fresh = self._client.fetch_market_trades(condition_id)
            if fresh:
                self._store.insert_trades(condition_id, fresh)
                trades = [
                    {"timestamp": int(t["timestamp"]), "asset": t.get("asset"),
                     "side": t.get("side"), "size": float(t.get("size", 0)),
                     "price": float(t.get("price", 0))}
                    for t in fresh if t.get("asset") == token_id
                ]
        if not trades:
            return None
        return reconstruct_bars(trades, asset=token_id)

    def _build_bars(self, market: Market):
        cached = self._store.fetch_trades(market.condition_id, asset=market.token_id)
        trades = [dict(r) for r in cached]
        if not trades:
            fresh = self._client.fetch_market_trades(market.condition_id)
            if fresh:
                self._store.insert_trades(market.condition_id, fresh)
                trades = [
                    {"timestamp": int(t["timestamp"]), "asset": t.get("asset"),
                     "side": t.get("side"), "size": float(t.get("size", 0)),
                     "price": float(t.get("price", 0))}
                    for t in fresh if t.get("asset") == market.token_id
                ]
        if not trades:
            return None
        return reconstruct_bars(trades, asset=market.token_id)
