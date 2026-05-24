"""Compare baseline LightGBM training across N markets vs strong regularization.

Used to evaluate how dataset size and regularization affect overfit
(train logloss vs val logloss gap) and held-out sharpe.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from config import ROOT
from src.ml.features import FEATURE_NAMES, iter_resolution_dataset
from src.ml.train_resolution import simulate_resolution_payoff, time_split

DB_PATH = str(ROOT / "data" / "traces.db")


def build_dataset():
    feats, labels, ts, mids = [], [], [], []
    for f, y, cid, _tid, t in iter_resolution_dataset(DB_PATH, min_trades=200):
        feats.append(f); labels.append(y); ts.append(t); mids.append(cid)
    return (
        np.array(feats, dtype=float),
        np.array(labels, dtype=int),
        np.array(ts, dtype=np.int64),
        mids,
    )


def run_config(name: str, X, y, ts, params: dict, num_boost: int = 500):
    train_idx, val_idx, test_idx = time_split(ts)
    train_data = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=FEATURE_NAMES,
                            reference=train_data)
    model = lgb.train(
        params, train_data, num_boost_round=num_boost,
        valid_sets=[train_data, val_data], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    train_loss = model.best_score["train"]["binary_logloss"]
    val_loss = model.best_score["val"]["binary_logloss"]
    test_prices = X[test_idx][:, 0]

    print(f"\n=== {name} ===")
    print(f"train logloss: {train_loss:.4f}  val logloss: {val_loss:.4f}  gap: {val_loss - train_loss:+.4f}")
    print(f"best iter: {model.best_iteration}")
    for thr in (0.04, 0.06, 0.10, 0.15):
        sim = simulate_resolution_payoff(model, X[test_idx], y[test_idx], test_prices, edge_threshold=thr)
        print(f"  edge>{thr}: n={sim.get('n_trades',0):4} "
              f"hit={sim.get('hit_rate',0):.3f} "
              f"avg={sim.get('avg_return',0):+.3f} "
              f"sharpe={sim.get('sharpe_per_trade',0):+.3f}")
    return {"name": name, "train_loss": train_loss, "val_loss": val_loss,
            "best_iter": model.best_iteration}


def main():
    X, y, ts, mids = build_dataset()
    print(f"Dataset: {len(X)} samples, {len(set(mids))} markets, pos_rate={y.mean():.3f}")

    configs = {
        "baseline": dict(objective="binary", metric="binary_logloss",
                         learning_rate=0.05, num_leaves=31,
                         feature_fraction=0.8, bagging_fraction=0.8,
                         bagging_freq=5, min_data_in_leaf=50, verbose=-1),
        "stronger_reg": dict(objective="binary", metric="binary_logloss",
                              learning_rate=0.03, num_leaves=15,
                              feature_fraction=0.6, bagging_fraction=0.6,
                              bagging_freq=3, min_data_in_leaf=200,
                              lambda_l2=1.0, verbose=-1),
        "small_trees": dict(objective="binary", metric="binary_logloss",
                             learning_rate=0.05, num_leaves=7,
                             feature_fraction=0.7, bagging_fraction=0.7,
                             bagging_freq=5, min_data_in_leaf=100,
                             lambda_l2=2.0, verbose=-1),
    }

    results = []
    for name, params in configs.items():
        results.append(run_config(name, X, y, ts, params))

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['name']:>14}: train={r['train_loss']:.3f} val={r['val_loss']:.3f} "
              f"gap={r['val_loss']-r['train_loss']:+.3f} iter={r['best_iter']}")


if __name__ == "__main__":
    main()
