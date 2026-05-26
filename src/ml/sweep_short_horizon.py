"""Sweep hyperparameters of the short-horizon strategy:
  - late_stage_days_max ∈ {3, 7, 14}
  - model type: classification (binary up/down) vs regression (predict return)
  - edge threshold

Goal: find the config with sharpe ≥ +0.5 per trade.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from config import ROOT
from src.ml.features import SHORT_HORIZON_FEATURE_NAMES, iter_short_horizon_dataset
from src.utils.logger import get_logger

log = get_logger("sweep_sh")
DB_PATH = str(ROOT / "data" / "traces.db")


def build_dataset(late_stage_days_max: float):
    feats, labels, fwd_rets, ts_list, mids = [], [], [], [], []
    for f, y, fwd, cid, _tid, t, _drop in iter_short_horizon_dataset(
        DB_PATH, min_trades=200, late_stage_days_max=late_stage_days_max,
    ):
        feats.append(f); labels.append(y); fwd_rets.append(fwd)
        ts_list.append(t); mids.append(cid)
    return (
        np.array(feats, dtype=float),
        np.array(labels, dtype=int),
        np.array(fwd_rets, dtype=float),
        np.array(ts_list, dtype=np.int64),
        mids,
    )


def time_split(ts):
    order = np.argsort(ts)
    n = len(ts)
    return order[:int(n*0.6)], order[int(n*0.6):int(n*0.8)], order[int(n*0.8):]


def train_classifier(X_tr, y_tr, X_vl, y_vl):
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=SHORT_HORIZON_FEATURE_NAMES)
    val_data = lgb.Dataset(X_vl, label=y_vl, feature_name=SHORT_HORIZON_FEATURE_NAMES,
                            reference=train_data)
    return lgb.train(
        {"objective": "binary", "metric": "binary_logloss",
         "learning_rate": 0.05, "num_leaves": 31,
         "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 5, "min_data_in_leaf": 50, "verbose": -1},
        train_data, num_boost_round=500,
        valid_sets=[train_data, val_data], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )


def train_regressor(X_tr, fwd_tr, X_vl, fwd_vl):
    train_data = lgb.Dataset(X_tr, label=fwd_tr, feature_name=SHORT_HORIZON_FEATURE_NAMES)
    val_data = lgb.Dataset(X_vl, label=fwd_vl, feature_name=SHORT_HORIZON_FEATURE_NAMES,
                            reference=train_data)
    return lgb.train(
        {"objective": "regression", "metric": "l2",
         "learning_rate": 0.05, "num_leaves": 31,
         "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 5, "min_data_in_leaf": 50, "verbose": -1},
        train_data, num_boost_round=500,
        valid_sets=[train_data, val_data], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )


def evaluate(p_pred, prices, fwd_rets, threshold_kind: str, thresholds: list[float]) -> list:
    """Return list of (threshold, n, hit, avg, std, sharpe) tuples.

    threshold_kind:
      - 'edge': trade when (p_pred - prices) > threshold (for classifier)
      - 'predicted_return': trade when p_pred > threshold (for regressor)
    """
    out = []
    for thr in thresholds:
        if threshold_kind == "edge":
            mask = (p_pred - prices) > thr
        else:
            mask = p_pred > thr
        if mask.sum() == 0:
            out.append((thr, 0, None, None, None, None))
            continue
        pnl = fwd_rets[mask]
        sharpe = pnl.mean() / (pnl.std() + 1e-9)
        out.append((thr, int(mask.sum()),
                    float((pnl > 0).mean()), float(pnl.mean()),
                    float(pnl.std()), float(sharpe)))
    return out


def main():
    overall: list[dict] = []
    for days_max in (3.0, 7.0, 14.0):
        print(f"\n{'='*70}\n  late_stage_days_max = {days_max}\n{'='*70}")
        X, y, fwd, ts, mids = build_dataset(days_max)
        if len(X) < 500:
            print("(too few samples, skipping)")
            continue
        print(f"Samples: {len(X)}, markets: {len(set(mids))}, pos rate: {y.mean():.3f}")
        train_idx, val_idx, test_idx = time_split(ts)
        prices_test = X[test_idx][:, 0]
        fwd_test = fwd[test_idx]

        # Classifier
        cls = train_classifier(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        p_pred = cls.predict(X[test_idx])
        print("\n  Classifier (edge = p_pred - price):")
        for thr, n, hit, avg, std, sh in evaluate(p_pred, prices_test, fwd_test,
                                                   "edge", [0.05, 0.10, 0.20, 0.30, 0.40]):
            if n == 0:
                continue
            print(f"    edge>{thr}: n={n:4} hit={hit:.3f} avg={avg:+.4f} "
                  f"std={std:.4f} sharpe={sh:+.3f}")
            overall.append({"days_max": days_max, "model": "classifier",
                            "threshold": thr, "n": n, "sharpe": sh, "avg": avg})

        # Regressor
        reg = train_regressor(X[train_idx], fwd[train_idx], X[val_idx], fwd[val_idx])
        rp = reg.predict(X[test_idx])
        print("\n  Regressor (predicted_return > threshold):")
        for thr, n, hit, avg, std, sh in evaluate(rp, prices_test, fwd_test,
                                                   "predicted_return", [0.01, 0.03, 0.05, 0.10, 0.20]):
            if n == 0:
                continue
            print(f"    pred>{thr}: n={n:4} hit={hit:.3f} avg={avg:+.4f} "
                  f"std={std:.4f} sharpe={sh:+.3f}")
            overall.append({"days_max": days_max, "model": "regressor",
                            "threshold": thr, "n": n, "sharpe": sh, "avg": avg})

    overall.sort(key=lambda x: x["sharpe"] or 0, reverse=True)
    print(f"\n{'='*70}\nTop 10 configs by sharpe (n ≥ 50):\n{'='*70}")
    shown = 0
    for r in overall:
        if r["n"] < 50:
            continue
        print(f"  days={r['days_max']:>4} {r['model']:>10} thr={r['threshold']:>5} "
              f"n={r['n']:5} avg={r['avg']:+.4f} sharpe={r['sharpe']:+.3f}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
