"""Numeric helpers for K-line / volume features."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def linear_regression(values: Sequence[float]) -> tuple[float, float]:
    """Return (slope, R^2) for y vs index."""
    if len(values) < 3:
        return 0.0, 0.0
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    sxy = ((x - x_mean) * (y - y_mean)).sum()
    sxx = ((x - x_mean) ** 2).sum()
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(slope), float(r2)


def safe_mean(values: Sequence[float]) -> float:
    if not len(values):
        return 0.0
    return float(np.mean(values))
