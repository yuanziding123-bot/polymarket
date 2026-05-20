"""Train LightGBM probability estimator on cached Polymarket trades.

Walk-forward split: earliest 60% of samples (by bar_ts) → train, next 20% → val, latest 20% → test.
Reports hit rate, sharpe, and feature importance on the holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from config import ROOT
from src.ml.features import FEATURE_NAMES, iter_dataset
from src.utils.logger import get_logger

log = get_logger("ml_train")

DB_PATH = str(ROOT / "data" / "traces.db")
MODEL_OUT = ROOT / "data" / "ml_model.txt"
META_OUT = ROOT / "data" / "ml_model_meta.json"


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feats_list, labels_list, fwds_list, market_ids, ts_list = [], [], [], [], []
    for feats, label, fwd, cid, ts in iter_dataset(DB_PATH, min_trades=200):
        feats_list.append(feats)
        labels_list.append(label)
        fwds_list.append(fwd)
        market_ids.append(cid)
        ts_list.append(ts)
    X = np.array(feats_list, dtype=float)
    y = np.array(labels_list, dtype=int)
    fwd_returns = np.array(fwds_list, dtype=float)
    ts = np.array(ts_list, dtype=np.int64)
    return X, y, fwd_returns, ts, market_ids


def time_split(ts: np.ndarray, train_frac: float = 0.6, val_frac: float = 0.2):
    order = np.argsort(ts)
    n = len(ts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    return train_idx, val_idx, test_idx


def evaluate(model, X_test, y_test, prices) -> dict:
    """Compute hit_rate / sharpe / edge-distribution on a held-out set."""
    p_pred = model.predict(X_test)
    # Treat predictions > 0.5 as "model thinks up". Hit if y_test == 1.
    preds_up = (p_pred > 0.5).astype(int)
    hit_rate = float((preds_up == y_test).mean())

    # Edge: ML prob - current market price (the feature at index 0)
    edges = p_pred - prices
    pnl_per_trial = np.where(y_test == 1, +1.0, -1.0) * np.sign(edges)

    sharpe = pnl_per_trial.mean() / (pnl_per_trial.std() + 1e-9)

    return {
        "n_samples": int(len(y_test)),
        "label_positive_rate": float(y_test.mean()),
        "model_pred_positive_rate": float(preds_up.mean()),
        "hit_rate": hit_rate,
        "directional_sharpe": float(sharpe),
        "edge_mean": float(edges.mean()),
        "edge_std": float(edges.std()),
        "edge_p25": float(np.percentile(edges, 25)),
        "edge_p75": float(np.percentile(edges, 75)),
    }


def trade_simulation(model, X_test, y_test, prices, forward_returns,
                     edge_threshold: float = 0.06) -> dict:
    """Simulate trading: long when model edge > threshold.

    Uses **real 24h forward return** (forward_returns) as PnL, not a binary
    win/lose proxy — this is honest sharpe.
    """
    p_pred = model.predict(X_test)
    edges = p_pred - prices
    mask = edges > edge_threshold
    if mask.sum() == 0:
        return {"n_trades": 0}
    pnl = forward_returns[mask]
    sharpe = pnl.mean() / (pnl.std() + 1e-9)
    return {
        "n_trades": int(mask.sum()),
        "trade_hit_rate": float((pnl > 0).mean()),
        "avg_return": float(pnl.mean()),
        "median_return": float(np.median(pnl)),
        "sharpe_per_trade": float(sharpe),
    }


def main() -> None:
    log.info("Building dataset from cached trades…")
    X, y, fwd_returns, ts, market_ids = build_dataset()
    log.info(f"Dataset: {len(X)} samples, {len(set(market_ids))} markets, "
             f"label pos rate={y.mean():.3f}")
    if len(X) < 500:
        log.error("Too few samples to train.")
        return

    train_idx, val_idx, test_idx = time_split(ts)
    log.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_data = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=FEATURE_NAMES,
                            reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }

    log.info("Training LightGBM…")
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[train_data, val_data], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    log.info("Evaluating on test set…")
    test_prices = X[test_idx][:, 0]   # 'price' is feature index 0
    eval_metrics = evaluate(model, X[test_idx], y[test_idx], test_prices)
    log.info(f"Directional eval: {json.dumps(eval_metrics, indent=2)}")

    test_fwd_returns = fwd_returns[test_idx]
    for thr in (0.02, 0.04, 0.06, 0.08, 0.10):
        sim = trade_simulation(model, X[test_idx], y[test_idx], test_prices,
                                test_fwd_returns, edge_threshold=thr)
        log.info(f"Edge>{thr}: {json.dumps(sim)}")

    # Feature importance
    importance = sorted(
        zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )[:15]
    print("\nTop 15 features by gain:")
    for name, imp in importance:
        print(f"  {name:>22}  {imp:.1f}")

    # Save
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    META_OUT.write_text(json.dumps({
        "n_samples": len(X), "n_markets": len(set(market_ids)),
        "test_metrics": eval_metrics,
        "feature_names": FEATURE_NAMES,
    }, indent=2))
    log.info(f"Saved model → {MODEL_OUT}")


if __name__ == "__main__":
    main()
