"""
All plotting functions used in the analysis.

Each function accepts an optional ``save_path``. When provided the figure
is written to disk at 300 dpi; when omitted the figure is only displayed
inline (useful while exploring in the notebook).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_and_show(save_path: Optional[str], close: bool = False) -> None:
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    if close:
        plt.close()


def _scatter_axes(
    ax,
    y_true: np.ndarray,
    y_pred_10f: np.ndarray,
    y_pred_mc: np.ndarray,
    title: Optional[str] = None,
    r2_mc: Optional[float] = None,
    r2_10f: Optional[float] = None,
) -> None:
    """Helper that draws a single scatter panel with identity + ±20% lines.

    ``r2_10f``/``r2_mc`` accept a precomputed mean-of-folds R² (matching the
    convention used everywhere else in this project); when omitted, falls
    back to a single pooled R² over all points shown in this panel.
    """
    r2_10f = r2_10f if r2_10f is not None else r2_score(y_true, y_pred_10f)
    r2_mc = r2_mc if r2_mc is not None else r2_score(y_true, y_pred_mc)

    max_val = max(np.max(y_true), np.max(y_pred_10f), np.max(y_pred_mc))
    x_line = np.linspace(0, max_val * 1.05, 200)

    ax.scatter(
        y_true, y_pred_10f,
        s=38, alpha=0.75,
        color="red", edgecolors="black",
        label=f"10-Fold CV (R²={r2_10f:.3f})",
    )
    ax.scatter(
        y_true, y_pred_mc,
        s=55, alpha=0.75,
        color="blue", marker="x",
        label=f"Monte Carlo Avg (R²={r2_mc:.3f})",
    )

    ax.plot(x_line, x_line, "k--", linewidth=1.8, label="Identity Line")
    ax.plot(x_line, 1.2 * x_line, "r--", linewidth=1.3, label="+20% Error")
    ax.plot(x_line, 0.8 * x_line, "r--", linewidth=1.3, label="-20% Error")

    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel("Experimental", fontsize=14)
    ax.set_ylabel("Predicted", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(fontsize=11, loc="upper left")


def _trend_axes(
    ax,
    y_true: np.ndarray,
    y_pred_10f: np.ndarray,
    y_pred_mc: np.ndarray,
    title: Optional[str] = None,
    r2_mc: Optional[float] = None,
    r2_10f: Optional[float] = None,
) -> None:
    """Helper that draws a sample-vs-strength trend panel.

    ``r2_10f``/``r2_mc`` accept a precomputed mean-of-folds R² (matching the
    convention used everywhere else in this project); when omitted, falls
    back to a single pooled R² over all points shown in this panel.
    """
    r2_10f = r2_10f if r2_10f is not None else r2_score(y_true, y_pred_10f)
    r2_mc = r2_mc if r2_mc is not None else r2_score(y_true, y_pred_mc)

    n = len(y_true)
    x = range(n)

    ax.plot(x, y_true, "ko-", label="Experimental", linewidth=1.2, markersize=3.8)
    ax.plot(
        x, y_pred_10f, color="red", linestyle="--", linewidth=1.8,
        label=f"Predicted (10-Fold CV R²={r2_10f:.3f})",
    )
    ax.plot(
        x, y_pred_mc, color="blue", linestyle="-.", linewidth=1.8,
        label=f"Predicted (Monte Carlo R²={r2_mc:.3f})",
    )

    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel("Sample Index", fontsize=14)
    ax.set_ylabel("Compressive Strength (MPa)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)


# ---------------------------------------------------------------------------
# Single-model plots
# ---------------------------------------------------------------------------


def plot_scatter(
    y_true,
    y_pred_10f,
    y_pred_mc,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    r2_mc: Optional[float] = None,
    r2_10f: Optional[float] = None,
) -> None:
    """Scatter plot of experimental vs predicted strength for a single model."""
    fig, ax = plt.subplots(figsize=(9, 8))
    _scatter_axes(
        ax,
        np.asarray(y_true).ravel(),
        np.asarray(y_pred_10f).ravel(),
        np.asarray(y_pred_mc).ravel(),
        title=title,
        r2_mc=r2_mc,
        r2_10f=r2_10f,
    )
    _save_and_show(save_path)


def plot_trend(
    y_true,
    y_pred_10f,
    y_pred_mc,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    r2_mc: Optional[float] = None,
    r2_10f: Optional[float] = None,
) -> None:
    """Sample-index trend plot for a single model."""
    fig, ax = plt.subplots(figsize=(14, 6))
    _trend_axes(
        ax,
        np.asarray(y_true).ravel(),
        np.asarray(y_pred_10f).ravel(),
        np.asarray(y_pred_mc).ravel(),
        title=title,
        r2_mc=r2_mc,
        r2_10f=r2_10f,
    )
    _save_and_show(save_path)


# ---------------------------------------------------------------------------
# Multi-model panels
# ---------------------------------------------------------------------------


def _panel_axes(fig):
    """Layout used for the 5-panel comparison grids: 2x2 + bottom-centered 5th."""
    gs = fig.add_gridspec(3, 4)
    return [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[2, 1:3]),
    ]


def _panel_axes_top3(fig):
    """Layout used for the 3-panel comparison grids: 2 top + 1 bottom-centered."""
    gs = fig.add_gridspec(2, 4)
    return [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 1:3]),
    ]


def plot_combined_trend(
    y_true,
    predictions_10fold: Dict[str, np.ndarray],
    predictions_mc: Dict[str, np.ndarray],
    mc_r2: Dict[str, float],
    model_order: List[str],
    save_path: Optional[str] = None,
    cv_r2: Optional[Dict[str, float]] = None,
) -> None:
    """5-panel trend plot, one panel per model."""
    y_true = np.asarray(y_true).ravel()
    fig = plt.figure(figsize=(18, 14))
    axes = _panel_axes(fig)
    for ax, name in zip(axes, model_order):
        _trend_axes(
            ax,
            y_true,
            predictions_10fold[name],
            predictions_mc[name],
            title=name,
            r2_mc=mc_r2.get(name),
            r2_10f=cv_r2.get(name) if cv_r2 is not None else None,
        )
    _save_and_show(save_path, close=True)


def plot_combined_scatter(
    y_true,
    predictions_10fold: Dict[str, np.ndarray],
    predictions_mc: Dict[str, np.ndarray],
    model_order: List[str],
    mc_r2: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    cv_r2: Optional[Dict[str, float]] = None,
) -> None:
    """5-panel scatter plot with identity and ±20% error lines, one per model."""
    y_true = np.asarray(y_true).ravel()
    fig = plt.figure(figsize=(18, 16))
    axes = _panel_axes(fig)
    for ax, name in zip(axes, model_order):
        _scatter_axes(
            ax,
            y_true,
            predictions_10fold[name],
            predictions_mc[name],
            title=name,
            r2_mc=mc_r2.get(name) if mc_r2 is not None else None,
            r2_10f=cv_r2.get(name) if cv_r2 is not None else None,
        )
    _save_and_show(save_path, close=True)


def plot_combined_top3_trend(
    y_true,
    predictions_10fold: Dict[str, np.ndarray],
    predictions_mc: Dict[str, np.ndarray],
    mc_r2: Dict[str, float],
    model_order: List[str],
    save_path: Optional[str] = None,
    cv_r2: Optional[Dict[str, float]] = None,
) -> None:
    """3-panel trend plot, one panel per model."""
    y_true = np.asarray(y_true).ravel()
    fig = plt.figure(figsize=(18, 10))
    axes = _panel_axes_top3(fig)
    for ax, name in zip(axes, model_order):
        _trend_axes(
            ax,
            y_true,
            predictions_10fold[name],
            predictions_mc[name],
            title=name,
            r2_mc=mc_r2.get(name),
            r2_10f=cv_r2.get(name) if cv_r2 is not None else None,
        )
    _save_and_show(save_path, close=True)


def plot_combined_top3_scatter(
    y_true,
    predictions_10fold: Dict[str, np.ndarray],
    predictions_mc: Dict[str, np.ndarray],
    model_order: List[str],
    mc_r2: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    cv_r2: Optional[Dict[str, float]] = None,
) -> None:
    """3-panel scatter plot with identity and ±20% error lines, one per model."""
    y_true = np.asarray(y_true).ravel()
    fig = plt.figure(figsize=(18, 11))
    axes = _panel_axes_top3(fig)
    for ax, name in zip(axes, model_order):
        _scatter_axes(
            ax,
            y_true,
            predictions_10fold[name],
            predictions_mc[name],
            title=name,
            r2_mc=mc_r2.get(name) if mc_r2 is not None else None,
            r2_10f=cv_r2.get(name) if cv_r2 is not None else None,
        )
    _save_and_show(save_path, close=True)


def plot_model_comparison_bars(
    pooled_metrics: pd.DataFrame,
    monte_carlo_metrics: pd.DataFrame,
    metric: str = "R2",
    save_path: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Side-by-side bars of a single metric for 10-fold CV vs Monte Carlo."""
    df = pooled_metrics.merge(
        monte_carlo_metrics,
        on="Model_ID",
        suffixes=("_10Fold", "_MC"),
    )
    # Preserve the order in pooled_metrics (already sorted by R2 desc)
    df = df.set_index("Model_ID").loc[pooled_metrics["Model_ID"]].reset_index()

    x = np.arange(len(df))
    bar_width = 0.36

    plt.figure(figsize=(11, 6))
    plt.bar(x - bar_width / 2, df[f"{metric}_10Fold"], width=bar_width, label=f"10-Fold CV {metric}")
    plt.bar(x + bar_width / 2, df[f"{metric}_MC"], width=bar_width, label=f"Monte Carlo {metric}")
    plt.xlabel("Model", fontsize=13)
    plt.ylabel(ylabel or metric, fontsize=13)
    plt.xticks(x, df["Model_ID"], rotation=0, fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    _save_and_show(save_path, close=True)


def plot_holdout_bars(
    holdout_metrics: pd.DataFrame,
    metric: str = "R2",
    save_path: Optional[str] = None,
) -> None:
    """Single-bar plot of any metric on the holdout test set."""
    plt.figure(figsize=(10, 6))
    plt.bar(holdout_metrics["Model_ID"], holdout_metrics[metric])
    plt.xlabel("Model", fontsize=13)
    plt.ylabel(metric, fontsize=13)
    plt.title(f"Model Comparison Based on Holdout Test {metric}", fontsize=14)
    plt.xticks(rotation=20, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    _save_and_show(save_path, close=True)


# ---------------------------------------------------------------------------
# Feature importance plots
# ---------------------------------------------------------------------------


def plot_feature_importance(
    fi_df: pd.DataFrame,
    save_path: Optional[str] = None,
    value_column: str = "Importance (%)",
    label_column: str = "Feature_Display",
) -> None:
    """Horizontal bar chart of individual feature importances."""
    plt.figure(figsize=(10, 6))
    plt.barh(fi_df[label_column], fi_df[value_column])
    plt.gca().invert_yaxis()
    plt.xlabel(value_column, fontsize=12)
    plt.ylabel("Mix Design Parameters", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    _save_and_show(save_path)


def plot_categorical_impact(
    cat_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar of per-category feature importance."""
    plt.figure(figsize=(10, 6))
    plt.barh(cat_df["Category"], cat_df["Importance (%)"])
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance Score (%)", fontsize=12)
    plt.ylabel("Mix Design Parameters", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    _save_and_show(save_path)


def plot_partial_dependence(
    model,
    X: np.ndarray,
    feature_indices: List[int],
    feature_labels: List[str],
    save_path: Optional[str] = None,
    grid_resolution: int = 80,
) -> None:
    """PDP + ICE plots for the given features."""
    from sklearn.inspection import partial_dependence
    import matplotlib.ticker as ticker

    TEAL         = "#1a6fa3"
    panel_labels = "abcdefgh"

    n = len(feature_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5), sharey=False)
    if n == 1:
        axes = [axes]

    # collect all data first to compute a shared y range
    all_data = []
    for feat_idx in feature_indices:
        pd_res = partial_dependence(model, X, features=[feat_idx], kind="average", grid_resolution=grid_resolution)
        all_data.append((pd_res["grid_values"][0], pd_res["average"][0]))

    all_avg_vals = np.concatenate([avg for _, avg in all_data])
    ylo = all_avg_vals.min() - 4
    yhi = all_avg_vals.max() + 4

    for i, (ax, feat_idx, xlabel) in enumerate(zip(axes, feature_indices, feature_labels)):
        grid, avg = all_data[i]

        ax.plot(grid, avg, color=TEAL, linewidth=2.5)
        ax.set_ylim(ylo, yhi)

        rug_y = ylo + (yhi - ylo) * 0.012
        ax.scatter(X[:, feat_idx], np.full(len(X), rug_y),
                   marker="|", color="#333333", s=80, alpha=0.7, linewidths=1.3, zorder=4)

        ax.text(0.03, 0.97, f"({panel_labels[i]})", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Predicted CS (MPa)", fontsize=12)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Partial Dependence — Nano Silica, Molarity and Binder Content",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    _save_and_show(save_path)


def plot_binder_pie(
    binder_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Pie chart of the relative importance of the three binders."""
    vals = binder_df["Importance (%)"].to_numpy()
    names = binder_df["Binder"].tolist()

    if vals.sum() == 0:
        # Nothing to show; produce an empty figure rather than failing
        vals = np.ones_like(vals)

    explode_idx = int(np.argmax(vals))
    explode = [0.05 if i == explode_idx else 0 for i in range(len(vals))]

    plt.figure(figsize=(8, 8))
    plt.pie(vals, labels=names, autopct="%1.1f%%", startangle=140, explode=explode)
    _save_and_show(save_path)
