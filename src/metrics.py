"""
Performance metrics used throughout the model evaluation.

The same metric set is used for 10-fold CV, Monte Carlo and holdout
evaluation, so it lives in one place to keep results comparable.

Metrics
-------
- R^2           : Coefficient of determination
- Pearson (r)   : Pearson correlation coefficient
- RMSE          : Root mean squared error
- MAE           : Mean absolute error
- MAPE          : Mean absolute percentage error (%)
- IoA           : Willmott's Index of Agreement
- Theta Mean    : Mean of the bias factor theta = y_true / y_pred
- Theta CoV     : Coefficient of variation of theta (COV-theta)
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

METRIC_COLUMNS: List[str] = [
    "R2",
    "Pearson (r)",
    "RMSE",
    "MAE",
    "MAPE",
    "IoA",
    "Theta Mean",
    "Theta CoV",
]


def index_of_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Willmott's Index of Agreement."""
    y_bar = np.mean(y_true)
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum((np.abs(y_pred - y_bar) + np.abs(y_true - y_bar)) ** 2)
    return 1.0 - numerator / denominator


MIN_VALID_PRED = 1e-6  # MPa; a prediction at or below this is physically meaningless


def theta_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Mean and coefficient of variation of theta = y_true / y_pred.

    Samples whose prediction is <= MIN_VALID_PRED are excluded: dividing by a
    near-zero prediction produces values of order 1e11 that swamp the mean and
    make the statistic meaningless. Returns NaN if fewer than two samples remain.
    Well-behaved models keep every sample, so this changes nothing for them.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    valid = y_pred > MIN_VALID_PRED
    if valid.sum() < 2:
        return float("nan"), float("nan")

    theta = y_true[valid] / y_pred[valid]
    theta_mean = float(np.mean(theta))
    theta_cov = float(np.std(theta, ddof=1) / theta_mean)
    return theta_mean, theta_cov


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """
    Compute the full metric suite for a single set of predictions.

    Returns a dict in the same order as ``METRIC_COLUMNS``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    r2 = float(r2_score(y_true, y_pred))
    r = float(np.corrcoef(y_pred, y_true)[0, 1])
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100.0
    ioa = float(index_of_agreement(y_true, y_pred))
    theta_mean, theta_cov = theta_stats(y_true, y_pred)

    return {
        "R2": r2,
        "Pearson (r)": r,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "IoA": ioa,
        "Theta Mean": theta_mean,
        "Theta CoV": theta_cov,
    }


def metrics_to_array(metrics: Dict[str, float]) -> np.ndarray:
    """Convert a metrics dict to an ordered numpy array."""
    return np.array([metrics[c] for c in METRIC_COLUMNS], dtype=float)


def metrics_table(rows: List[Dict[str, float]], model_names: List[str]) -> pd.DataFrame:
    """Combine a list of per-model metric dicts into a sorted DataFrame."""
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    df.insert(0, "Model_ID", model_names)
    return df.sort_values("R2", ascending=False).reset_index(drop=True)


def fold_mean_table(
    fold_metrics_list: List[List[Dict[str, float]]],
    model_names: List[str],
) -> pd.DataFrame:
    """
    Compute per-metric MEAN across folds/repeats for each model.

    Each metric is calculated once per fold/repeat (see ``fold_metrics``),
    then averaged — this is the mean-over-folds convention, not a single
    metric computed on all pooled out-of-fold predictions at once.

    Parameters
    ----------
    fold_metrics_list : one list of fold-metric dicts per model, in the same
                        order as ``model_names``.
    """
    rows = []
    for name, fold_metrics in zip(model_names, fold_metrics_list):
        arr = np.vstack([metrics_to_array(m) for m in fold_metrics])
        mean = arr.mean(axis=0)
        rows.append({col: float(v) for col, v in zip(METRIC_COLUMNS, mean)})
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    df.insert(0, "Model_ID", model_names)
    return df.sort_values("R2", ascending=False).reset_index(drop=True)


def fold_std_table(
    fold_metrics_list: List[List[Dict[str, float]]],
    model_names: List[str],
    sort_order: List[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-metric SD across folds/repeats for each model.

    Parameters
    ----------
    fold_metrics_list : one list of fold-metric dicts per model, in the same
                        order as ``model_names``.
    sort_order : if provided, reindex rows to match this model order (e.g. the
                 order already used in the paired mean table).
    """
    rows = []
    for name, fold_metrics in zip(model_names, fold_metrics_list):
        arr = np.vstack([metrics_to_array(m) for m in fold_metrics])
        std = arr.std(axis=0, ddof=1)
        rows.append({col: float(v) for col, v in zip(METRIC_COLUMNS, std)})
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    df.insert(0, "Model_ID", model_names)
    if sort_order is not None:
        df = df.set_index("Model_ID").reindex(sort_order).reset_index()
    return df


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    """Pretty-print a metric dict on two lines."""
    print(f"--- {label} ---")
    print(
        f"R2: {metrics['R2']:.3f} | "
        f"r: {metrics['Pearson (r)']:.3f} | "
        f"RMSE: {metrics['RMSE']:.3f} | "
        f"MAE: {metrics['MAE']:.3f} | "
        f"MAPE: {metrics['MAPE']:.2f}%"
    )
    print(
        f"IoA: {metrics['IoA']:.3f} | "
        f"Theta Mean: {metrics['Theta Mean']:.3f} | "
        f"Theta CoV: {metrics['Theta CoV']:.3f}"
    )
