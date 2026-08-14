"""
Scatter and Trend plots (Experimental vs Predicted) for the four real
200-trial-tuned models (Extra Trees, Random Forest, CatBoost, XGBoost --
Linear Regression wasn't part of the tuning search), using their actual
tuned hyperparameters from tuned200_summary.pkl.

DATA_PATH must be "Optimization - NewDataset (clean, all rows).xlsx" --
the exact dataset the 200-trial tuning run used (missing NS size(nm)
values median-filled). An earlier version of this script pointed at
"Optimization - NewDataset (clean, NS30).xlsx" (missing values filled
with a fixed 30nm instead), a later/different dataset version that only
differs on 4 of 136 rows but was enough to drop XGBoost's recomputed R2
from ~0.92 to 0.907 against an official 0.925 -- not a bug in the CV
code, just the wrong file.

The R2 shown on these plots is computed to match Results_Summary_Tuned200.xlsx
EXACTLY (all four models, both 10-Fold and Monte Carlo, verified to 6 decimal
places): 10-Fold R2 uses the POOLED metric (one R2 over all 136 out-of-fold
predictions at once, via cv.metrics["R2"]) -- NOT the mean-of-fold average
that src/training.py uses elsewhere in this project. Monte Carlo R2 already
matched by construction (mc.avg_metrics["R2"] is exactly what tune_200.py
stored as the official number, so no change was needed there). Per-sample
predictions still can't be pulled from tuned200_summary.pkl (it only stores
aggregate metrics), so this script reruns CV itself to get them -- but on
the correct dataset with the correct (pooled) aggregation, that rerun's
summary R2 reproduces the official numbers exactly, even though the
underlying per-sample predictions are a fresh evaluation.

(Results_Summary_Tuned200.xlsx also has separate "10Fold_Mean±SD" /
"MonteCarlo_Mean±SD" sheets reporting mean-of-fold/repeat ± standard
deviation -- a deliberately different, also-legitimate view of the same CV
run, not plotted here. See run_tuned_summary.py's docstring.)

Both plots (scatter + trend) are drawn from ONE computation here, so their
R2 values are guaranteed to match each other exactly.

Usage (from an activated venv, run from the Good version directory so 'src'
resolves):
    python run_tuned_plots.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from src.data import load_data
from src.plots import _scatter_axes, _trend_axes
from src.validation import kfold_cv, monte_carlo_cv

HERE = Path(__file__).parent
DATA_PATH = HERE / "Optimization - NewDataset (clean, all rows).xlsx"
SUMMARY_PATH = HERE / "tuned200_summary.pkl"
SCATTER_OUT = HERE / "Scatter_TunedModels.png"
TREND_OUT = HERE / "Trend_TunedModels.png"

MODEL_ORDER = ["XGBoost", "CatBoost", "Extra Trees", "Random Forest"]

SUBTITLE = (
    "10-Fold CV (pooled) and Monte Carlo CV, 100 repeats -- matches\n"
    "Results_Summary_Tuned200.xlsx (the official tuning-time numbers) exactly"
)


def main() -> None:
    dataset = load_data(file_path=str(DATA_PATH), extra_features=["NS size(nm)"])
    print(dataset.summary())
    X, y = dataset.X, dataset.y

    summary = joblib.load(SUMMARY_PATH)
    tuned_models = summary["tuned_models"]

    per_model = {}
    for name in MODEL_ORDER:
        print(f"\n=== {name} ===")
        template = tuned_models[name]

        cv = kfold_cv(template, X, y, n_splits=10, random_state=42)
        mc = monte_carlo_cv(
            template, X, y, n_splits=100, test_size=0.20, random_state=42,
            fallback_predictions=cv.predictions,
        )
        r2_10f = cv.metrics["R2"]
        r2_mc = mc.avg_metrics["R2"]
        print(f"  10-Fold R2 (pooled)            : {r2_10f:.4f}")
        print(f"  Monte Carlo R2 (mean-of-repeat): {r2_mc:.4f}")

        per_model[name] = (cv, mc, r2_10f, r2_mc)

    # ----------------------------- Scatter plot -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.ravel()
    for ax, name in zip(axes, MODEL_ORDER):
        cv, mc, r2_10f, r2_mc = per_model[name]
        _scatter_axes(ax, y.to_numpy(), cv.predictions, mc.sample_predictions,
                       title=name, r2_10f=r2_10f, r2_mc=r2_mc)
    fig.suptitle(
        f"Actual vs. Predicted 28-Day Compressive Strength — Tuned Models\n{SUBTITLE}",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(SCATTER_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {SCATTER_OUT}")

    # ----------------------------- Trend plot --------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.ravel()
    for ax, name in zip(axes, MODEL_ORDER):
        cv, mc, r2_10f, r2_mc = per_model[name]
        _trend_axes(ax, y.to_numpy(), cv.predictions, mc.sample_predictions,
                    title=name, r2_10f=r2_10f, r2_mc=r2_mc)
    fig.suptitle(
        f"Experimental vs. Predicted 28-Day Compressive Strength — Tuned Models (Trend)\n{SUBTITLE}",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(TREND_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {TREND_OUT}")


if __name__ == "__main__":
    main()
