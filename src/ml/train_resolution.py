"""Train LightGBM on RESOLVED Polymarket markets with real outcome as label.

Difference from train.py:
- Label = 1 if THIS specific token resolved to $1 (i.e. event truly went its way)
- Bactest uses true resolution payoff:
    win  → token went from entry → $1.00 → return = (1 - entry) / entry
    lose → token went from entry → $0.00 → return = -1.0

This is the proper probability-arbitrage backtest.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from config import ROOT
from src.ml.features import FEATURE_NAMES, iter_resolution_dataset
from src.utils.logger import get_logger

log = get_logger("ml_train_res")

DB_PATH = str(ROOT / "data" / "traces.db")
MODEL_OUT = ROOT / "data" / "ml_model_resolution.txt"
META_OUT = ROOT / "data" / "ml_model_resolution_meta.json"


def build_dataset():
    feats_list, labels_list, market_ids, token_ids, ts_list = [], [], [], [], []
    for feats, label, cid, tid, ts in iter_resolution_dataset(DB_PATH, min_trades=200):
        feats_list.append(feats)
        labels_list.append(label)
        market_ids.append(cid)
        token_ids.append(tid)
        ts_list.append(ts)
    X = np.array(feats_list, dtype=float)
    y = np.array(labels_list, dtype=int)
    ts = np.array(ts_list, dtype=np.int64)
    return X, y, ts, market_ids, token_ids


def time_split(ts, train_frac=0.6, val_frac=0.2):
    order = np.argsort(ts)
    n = len(ts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def simulate_resolution_payoff(model, X, y, prices, edge_threshold: float) -> dict:
    """Real probability-arbitrage payoff: win → (1-price)/price, lose → -1."""
    p_pred = model.predict(X)
    edges = p_pred - prices
    mask = edges > edge_threshold
    if mask.sum() == 0:
        return {"n_trades": 0}
    # Per-trade PnL: if token actually won (y=1), gain = (1-price)/price; else lose = -1.
    win_pnl = (1.0 - prices) / np.clip(prices, 1e-6, 1.0)
    pnl = np.where(y == 1, win_pnl, -1.0)
    selected = pnl[mask]
    sharpe = selected.mean() / (selected.std() + 1e-9)
    return {
        "n_trades": int(mask.sum()),
        "hit_rate": float(y[mask].mean()),
        "avg_return": float(selected.mean()),
        "median_return": float(np.median(selected)),
        "win_count": int(y[mask].sum()),
        "loss_count": int(mask.sum() - y[mask].sum()),
        "sharpe_per_trade": float(sharpe),
        "max_win": float(selected.max()) if len(selected) else 0.0,
        "max_loss": float(selected.min()) if len(selected) else 0.0,
    }


def main():
    log.info("Building resolution dataset…")
    X, y, ts, market_ids, token_ids = build_dataset()
    log.info(f"Dataset: {len(X)} samples, {len(set(market_ids))} markets, "
             f"label pos rate (=winning side)={y.mean():.3f}")
    if len(X) < 500:
        log.error("Too few samples to train.")
        return

    train_idx, val_idx, test_idx = time_split(ts)
    log.info(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

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

    # Evaluate
    test_prices = X[test_idx][:, 0]
    print("\n=== Resolution backtest ===")
    print(f"Test set: {len(test_idx)} samples ({y[test_idx].mean():.3f} positive)")
    for thr in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
        sim = simulate_resolution_payoff(model, X[test_idx], y[test_idx], test_prices, edge_threshold=thr)
        print(f"  edge>{thr}: {json.dumps(sim)}")

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
        "n_samples": len(X),
        "n_markets": len(set(market_ids)),
        "feature_names": FEATURE_NAMES,
        "label_pos_rate": float(y.mean()),
    }, indent=2))
    log.info(f"Saved model → {MODEL_OUT}")


if __name__ == "__main__":
    main()
