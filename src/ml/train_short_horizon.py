"""Train LightGBM to predict 6h-forward price movement in late-stage markets.

Strategy B per mentor: rather than estimating final resolution probability
(weak alpha, ~consensus-following), predict short-term price moves in the
final 14 days of a market's life when prices are physically converging
toward 0 or 1.

Label: 1 if price rises >1% in 6h, else 0.
Feature: 33 original + days_to_resolution + log_days_to_resolution.
Filter: only sample when days_to_resolution ≤ 14.

Backtest PnL: real 6h forward return per trade.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from config import ROOT
from src.ml.features import SHORT_HORIZON_FEATURE_NAMES, iter_short_horizon_dataset
from src.utils.logger import get_logger

log = get_logger("ml_train_sh")

DB_PATH = str(ROOT / "data" / "traces.db")
MODEL_OUT = ROOT / "data" / "ml_model_short_horizon.txt"
MODEL_REG_OUT = ROOT / "data" / "ml_model_short_horizon_regressor.txt"
META_OUT = ROOT / "data" / "ml_model_short_horizon_meta.json"


def build_dataset(late_stage_days_max: float = 14.0):
    feats, labels, fwd_rets, mids, tids, ts_list = [], [], [], [], [], []
    for f, y, fwd, cid, tid, t in iter_short_horizon_dataset(
        DB_PATH, min_trades=200, late_stage_days_max=late_stage_days_max,
    ):
        feats.append(f); labels.append(y); fwd_rets.append(fwd)
        mids.append(cid); tids.append(tid); ts_list.append(t)
    return (
        np.array(feats, dtype=float),
        np.array(labels, dtype=int),
        np.array(fwd_rets, dtype=float),
        np.array(ts_list, dtype=np.int64),
        mids,
    )


def time_split(ts, train_frac=0.6, val_frac=0.2):
    order = np.argsort(ts)
    n = len(ts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def simulate_short_horizon(model, X, y, prices, fwd_rets, edge_threshold: float) -> dict:
    """Buy when model edge > threshold; PnL = actual 6h forward return."""
    p_pred = model.predict(X)
    edges = p_pred - prices
    mask = edges > edge_threshold
    if mask.sum() == 0:
        return {"n_trades": 0}
    pnl = fwd_rets[mask]
    sharpe = pnl.mean() / (pnl.std() + 1e-9)
    return {
        "n_trades": int(mask.sum()),
        "trade_hit_rate": float((pnl > 0).mean()),
        "avg_return": float(pnl.mean()),
        "median_return": float(np.median(pnl)),
        "std_return": float(pnl.std()),
        "sharpe_per_trade": float(sharpe),
        "max_win": float(pnl.max()),
        "max_loss": float(pnl.min()),
    }


def main(late_stage_days_max: float = 14.0):
    log.info(f"Building short-horizon dataset (late_stage_days_max={late_stage_days_max})…")
    X, y, fwd_rets, ts, mids = build_dataset(late_stage_days_max)
    log.info(f"Dataset: {len(X)} samples, {len(set(mids))} markets, label pos rate={y.mean():.3f}")
    if len(X) < 500:
        log.error("Too few samples to train.")
        return

    train_idx, val_idx, test_idx = time_split(ts)
    log.info(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_data = lgb.Dataset(X[train_idx], label=y[train_idx],
                              feature_name=SHORT_HORIZON_FEATURE_NAMES)
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx],
                            feature_name=SHORT_HORIZON_FEATURE_NAMES,
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

    # Eval
    print(f"\n=== Short-horizon backtest (6h forward, late-stage ≤ {late_stage_days_max}d) ===")
    print(f"Test set: {len(test_idx)} samples, label pos rate={y[test_idx].mean():.3f}")
    test_prices = X[test_idx][:, 0]
    test_fwd = fwd_rets[test_idx]
    for thr in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
        sim = simulate_short_horizon(model, X[test_idx], y[test_idx], test_prices, test_fwd,
                                      edge_threshold=thr)
        if sim.get("n_trades", 0):
            print(f"  edge>{thr}: n={sim['n_trades']:5} hit={sim['trade_hit_rate']:.3f} "
                  f"avg={sim['avg_return']:+.4f} std={sim['std_return']:.4f} "
                  f"sharpe={sim['sharpe_per_trade']:+.3f}")
        else:
            print(f"  edge>{thr}: n=0")

    # Feature importance
    importance = sorted(
        zip(SHORT_HORIZON_FEATURE_NAMES, model.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )[:15]
    print("\nTop 15 features by gain:")
    for name, imp in importance:
        print(f"  {name:>26}  {imp:.1f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))

    # Also train regressor (predicts actual 6h forward return) — this gave
    # the best per-trade sharpe (+0.198) in the sweep at pred>0.2 threshold.
    log.info("Training regressor (predict 6h forward return)…")
    train_data_r = lgb.Dataset(X[train_idx], label=fwd_rets[train_idx],
                                feature_name=SHORT_HORIZON_FEATURE_NAMES)
    val_data_r = lgb.Dataset(X[val_idx], label=fwd_rets[val_idx],
                              feature_name=SHORT_HORIZON_FEATURE_NAMES,
                              reference=train_data_r)
    regressor = lgb.train(
        {"objective": "regression", "metric": "l2",
         "learning_rate": 0.05, "num_leaves": 31,
         "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 5, "min_data_in_leaf": 50, "verbose": -1},
        train_data_r, num_boost_round=500,
        valid_sets=[train_data_r, val_data_r], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    regressor.save_model(str(MODEL_REG_OUT))
    log.info(f"Saved regressor → {MODEL_REG_OUT}")

    # Evaluate regressor too
    rp = regressor.predict(X[test_idx])
    print("\nRegressor backtest (pred = predicted 6h return):")
    for thr in (0.05, 0.10, 0.15, 0.20):
        mask = rp > thr
        if mask.sum() == 0:
            continue
        pnl = test_fwd[mask]
        sharpe = pnl.mean() / (pnl.std() + 1e-9)
        print(f"  pred>{thr}: n={int(mask.sum()):4} hit={float((pnl>0).mean()):.3f} "
              f"avg={float(pnl.mean()):+.4f} std={float(pnl.std()):.4f} "
              f"sharpe={float(sharpe):+.3f}")

    META_OUT.write_text(json.dumps({
        "n_samples": len(X),
        "n_markets": len(set(mids)),
        "feature_names": SHORT_HORIZON_FEATURE_NAMES,
        "late_stage_days_max": late_stage_days_max,
        "label_pos_rate": float(y.mean()),
        "model_type": "regressor",
        "decision_rule": "predicted_return > 0.20",
    }, indent=2))
    log.info(f"Saved meta → {META_OUT}")


if __name__ == "__main__":
    main()
