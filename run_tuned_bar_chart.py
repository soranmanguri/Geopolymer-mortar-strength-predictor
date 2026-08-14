"""
Bar chart comparing the four tuned models' 10-Fold CV vs Monte Carlo R2,
built directly from tuned200_summary.pkl's stored tuned_cv/tuned_mc values
-- the SAME numbers as Results_Summary_Tuned200.xlsx, with zero
recomputation. Unlike the scatter/trend plots, a bar chart only needs the
already-stored aggregate metric per model, not per-sample predictions, so
this is guaranteed to match the xlsx exactly.

Usage (from an activated venv, run from the Good version directory so 'src'
resolves):
    python run_tuned_bar_chart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: savefig works, plt.show() is a no-op instead of blocking

import joblib
import pandas as pd

from src.plots import plot_model_comparison_bars

HERE = Path(__file__).parent
SUMMARY_PATH = HERE / "tuned200_summary.pkl"
SAVE_PATH = HERE / "Bar_TunedModels_R2.png"


def main() -> None:
    summary = joblib.load(SUMMARY_PATH)

    cv_df = pd.DataFrame(summary["tuned_cv"]).T.reset_index().rename(columns={"index": "Model_ID"})
    mc_df = pd.DataFrame(summary["tuned_mc"]).T.reset_index().rename(columns={"index": "Model_ID"})
    cv_df = cv_df.sort_values("R2", ascending=False).reset_index(drop=True)

    plot_model_comparison_bars(
        pooled_metrics=cv_df,
        monte_carlo_metrics=mc_df,
        metric="R2",
        ylabel="R²",
        save_path=str(SAVE_PATH),
    )
    print(f"Saved: {SAVE_PATH}")
    print("\nValues plotted (identical to Results_Summary_Tuned200.xlsx):")
    print(cv_df[["Model_ID", "R2"]].rename(columns={"R2": "10Fold_R2"}).to_string(index=False))
    print(mc_df[["Model_ID", "R2"]].rename(columns={"R2": "MonteCarlo_R2"}).to_string(index=False))


if __name__ == "__main__":
    main()
