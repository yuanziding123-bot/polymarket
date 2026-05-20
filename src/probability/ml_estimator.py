"""ML-based probability estimator. Drop-in replacement for ProbabilityEstimator
that uses a trained LightGBM model on price + volume + trade-flow features
instead of LLM + news fusion.

Same interface as ProbabilityEstimator so the rest of the pipeline doesn't care:
    estimator.estimate(market, detection) -> ProbabilityEstimate
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from config import ROOT
from src.data.polymarket_client import PolymarketClient
from src.data.types import DetectionResult, Market, ProbabilityEstimate
from src.ml.features import (
    FEATURE_NAMES,
    MIN_HISTORY_BARS,
    extract_features,
    reconstruct_bars,
)
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("ml_estimator")

DEFAULT_MODEL_PATH = ROOT / "data" / "ml_model.txt"


class MLProbabilityEstimator:
    def __init__(
        self,
        client: PolymarketClient,
        store: TraceStore | None = None,
        model_path: Path | None = None,
    ) -> None:
        path = model_path or DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ML model not found at {path}. Run `python -m src.ml.train` first."
            )
        self._model = lgb.Booster(model_file=str(path))
        self._client = client
        self._store = store or TraceStore()
        log.info(f"Loaded ML model from {path}")

    def estimate(
        self,
        market: Market,
        detection: DetectionResult,
        correlated_price: float | None = None,
    ) -> ProbabilityEstimate:
        """Predict P(price up >1% in 24h) for this market and return as p_true."""
        bars = self._build_bars(market)
        if bars is None or len(bars) < MIN_HISTORY_BARS:
            return ProbabilityEstimate(
                p_true=market.price,
                components={"ml": market.price, "fallback": 1.0},
                reasoning=["Not enough history; fell back to market price"],
                confidence="low",
                uncertainty="insufficient_history",
            )

        i = len(bars) - 1
        features = extract_features(bars, i)
        if features is None:
            return ProbabilityEstimate(
                p_true=market.price,
                components={"ml": market.price},
                reasoning=["Feature extraction failed"],
                confidence="low",
                uncertainty="feature_error",
            )

        p_pred = float(self._model.predict(features.reshape(1, -1))[0])
        edge = p_pred - market.price

        # Confidence heuristic: bigger absolute edge → higher confidence.
        # This is a placeholder; ideally we'd calibrate the model's uncertainty.
        abs_edge = abs(edge)
        if abs_edge < 0.03:
            confidence = "low"
        elif abs_edge < 0.08:
            confidence = "medium"
        else:
            confidence = "high"

        return ProbabilityEstimate(
            p_true=p_pred,
            components={"ml": p_pred, "market": market.price},
            reasoning=[
                f"LightGBM 33-feature model on {len(bars)} bars of history",
                f"Top features by training importance: dd_60, dd_120, price, ma_60",
                f"Edge = {edge:+.4f}",
            ],
            confidence=confidence,
            uncertainty=f"baseline_proxy_label_24h_forward_return_n_train=34k",
        )

    def _build_bars(self, market: Market):
        """Fetch fresh trades for this market and bucket into hourly bars.

        Uses cache first; only hits API for new trades since last cache entry.
        """
        cached = self._store.fetch_trades(market.condition_id, asset=market.token_id)
        # Convert cached rows to dicts for reconstruct_bars
        trades = [dict(r) for r in cached]
        if not trades:
            # No cache → fetch fresh
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
