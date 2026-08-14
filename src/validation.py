"""
Validation routines: 10-fold cross-validation, Monte Carlo simulation
and a single train/test holdout split.

Each routine returns a uniform structure so downstream code (plots,
result tables, summary spreadsheets) can treat them interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    train_test_split,
)

from .metrics import compute_metrics, metrics_to_array


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Result of a single cross-validation run."""

    predictions: np.ndarray
    metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class MonteCarloResult:
    """Result of an N-repeat Monte Carlo evaluation."""

    avg_metrics: Dict[str, float]
    sample_predictions: np.ndarray         # sample-wise mean prediction across repeats
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 10-fold CV
# ---------------------------------------------------------------------------


def kfold_cv(
    model,
    X,
    y,
    n_splits: int = 10,
    random_state: int = 42,
) -> CVResult:
    """
    Run K-fold cross-validation and return out-of-fold predictions and metrics.

    Iterates folds manually so per-fold metrics are captured alongside the
    pooled metrics computed on all out-of-fold predictions at once.
    The caller's model instance is never mutated (clone is used per fold).
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    y_arr = np.asarray(y).ravel()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_pred_all = np.empty(len(y_arr), dtype=float)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, test_idx in kf.split(np.arange(len(y_arr))):
        if is_dataframe:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_arr = np.asarray(X)
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pr = m.predict(X_te).ravel()

        y_pred_all[test_idx] = y_pr
        fold_metrics.append(compute_metrics(y_te, y_pr))

    metrics = compute_metrics(y_arr, y_pred_all)
    return CVResult(predictions=y_pred_all, metrics=metrics, fold_metrics=fold_metrics)


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def monte_carlo_cv(
    model,
    X,
    y,
    n_splits: int = 100,
    test_size: float = 0.20,
    random_state: int = 42,
    fallback_predictions: np.ndarray | None = None,
) -> MonteCarloResult:
    """
    Repeat a random train/test split N times.

    For each repeat the metric set is computed and stored, then averaged.
    Predictions are also accumulated per sample so a "Monte Carlo average
    prediction" series can be plotted next to the experimental values.

    Parameters
    ----------
    fallback_predictions : optional ndarray
        If provided, any sample that was never selected for testing across
        all repeats is filled with this fallback (typically 10-fold CV
        predictions). Without a fallback, untested samples remain NaN.
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    y_series = pd.Series(np.asarray(y).ravel(), index=X.index if is_dataframe else None)

    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    fold_metrics: List[Dict[str, float]] = []
    pred_sum = np.zeros(len(X), dtype=float)
    pred_count = np.zeros(len(X), dtype=int)

    # Iterate by integer position so it works with both DataFrames and ndarrays
    indices = np.arange(len(X))
    for train_idx, test_idx in rs.split(indices):
        if is_dataframe:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_series.iloc[train_idx], y_series.iloc[test_idx]
        else:
            X_arr = np.asarray(X)
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            y_tr, y_te = y_series.iloc[train_idx], y_series.iloc[test_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pr = m.predict(X_te).ravel()

        fold_metrics.append(compute_metrics(y_te.to_numpy(), y_pr))
        pred_sum[test_idx] += y_pr
        pred_count[test_idx] += 1

    # Average per-fold metrics
    arr = np.vstack([metrics_to_array(m) for m in fold_metrics])
    avg = arr.mean(axis=0)
    from .metrics import METRIC_COLUMNS  # local import to avoid cycle
    avg_metrics = {col: float(v) for col, v in zip(METRIC_COLUMNS, avg)}

    # Sample-wise mean prediction
    sample_predictions = np.divide(
        pred_sum,
        pred_count,
        out=np.full(len(X), np.nan, dtype=float),
        where=pred_count != 0,
    )

    if fallback_predictions is not None:
        missing = np.isnan(sample_predictions)
        sample_predictions[missing] = np.asarray(fallback_predictions)[missing]

    return MonteCarloResult(
        avg_metrics=avg_metrics,
        sample_predictions=sample_predictions,
        fold_metrics=fold_metrics,
    )


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------


def holdout_evaluation(
    model,
    X,
    y,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train on a random 1-test_size portion of the data, evaluate on the rest.

    Returns
    -------
    (metrics, y_test, y_pred)
        The metric dict and the actual / predicted vectors on the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    m = clone(model)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test).ravel()
    metrics = compute_metrics(np.asarray(y_test).ravel(), y_pred)
    return metrics, np.asarray(y_test).ravel(), y_pred
