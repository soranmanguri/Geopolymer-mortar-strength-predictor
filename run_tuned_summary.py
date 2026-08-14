"""
Saves Results_Summary_Tuned200.xlsx from the OFFICIAL numbers stored in
tuned200_summary.pkl (tuned_cv / tuned_mc) at the time the 200-trial Optuna
search was run, PLUS standard deviation across folds/repeats.

The "10Fold_CV" / "MonteCarlo_CV" sheets do NOT recompute anything -- they
only read what's already saved, so those numbers are exactly what the tuning
process itself produced (e.g. XGBoost 10-Fold R2 = 0.925, POOLED over all
136 out-of-fold predictions at once).

tuned200_summary.pkl only stored the final aggregate metric per model, not
the per-fold/per-repeat values SD needs -- those were never saved. So the
"10Fold_Mean±SD" / "MonteCarlo_Mean±SD" sheets ARE a recompute: same tuned
models, same dataset, same random_state=42 as everywhere else in this
project (verified to reproduce the official R2 to 15 decimal places -- see
run_tuned_plots.py's docstring). This mirrors exactly how
Results_Baseline\\Results_Summary.xlsx presents baseline SD (via
src/metrics.py's fold_mean_table/fold_std_table), so tuned and baseline
results are directly comparable.

One consequence worth flagging: fold_mean_table's "R2" is the MEAN of the
10 individual fold R2 values (~0.920 for XGBoost), not the pooled R2 (0.925)
in the "10Fold_CV" sheet above it -- mean-of-fold and pooled are two
different, both-legitimate aggregations of the same CV run (see
run_tuned_plots.py's docstring for why they differ). The SD sheets exist to
show spread across folds, and mean-of-fold is the aggregation SD naturally
pairs with (pairing an SD-of-folds with a pooled mean would mix two
different units of analysis). This is a deliberate choice, not an
inconsistency to fix -- the "10Fold_CV" sheet stays the official,
untouched, stored-at-tuning-time number.

Usage (from an activated venv; can be run from anywhere, 'src' resolves via
this script's own directory):
    python run_tuned_summary.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import load_data
from src.metrics import fold_mean_table, fold_std_table
from src.validation import kfold_cv, monte_carlo_cv

HERE = Path(__file__).parent
DATA_PATH = HERE / "Optimization - NewDataset (clean, all rows).xlsx"
SUMMARY_PATH = HERE / "tuned200_summary.pkl"
OUT_PATH = HERE / "Results_Summary_Tuned200.xlsx"

METRIC_COLS = ["R2", "Pearson (r)", "RMSE", "MAE", "IoA", "Theta Mean", "Theta CoV"]
MODEL_ORDER = ["XGBoost", "CatBoost", "Extra Trees", "Random Forest"]


def _fmt_mean_sd(mean_df: pd.DataFrame, sd_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, mrow in mean_df.iterrows():
        name = mrow["Model_ID"]
        srow = sd_df[sd_df["Model_ID"] == name].iloc[0]
        entry = {"Model_ID": name}
        for m in METRIC_COLS:
            entry[m] = f"{mrow[m]:.3f} ± {srow[m]:.3f}"
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    summary = joblib.load(SUMMARY_PATH)

    # ------------------------- official, stored numbers -------------------
    cv_df = pd.DataFrame(summary["tuned_cv"]).T[METRIC_COLS]
    mc_df = pd.DataFrame(summary["tuned_mc"]).T[METRIC_COLS]
    cv_df.index.name = "Model_ID"
    mc_df.index.name = "Model_ID"
    cv_df = cv_df.sort_values("R2", ascending=False).reset_index()
    mc_df = mc_df.sort_values("R2", ascending=False).reset_index()

    params_df = pd.DataFrame(summary["tuned_params"]).T
    params_df.index.name = "Model_ID"
    params_df = params_df.reset_index()

    # ------------------------- recompute for SD -----------------------------
    dataset = load_data(file_path=str(DATA_PATH), extra_features=["NS size(nm)"])
    X, y = dataset.X, dataset.y
    tuned_models = summary["tuned_models"]

    cv_fold_metrics = []
    mc_fold_metrics = []
    for name in MODEL_ORDER:
        model = tuned_models[name]
        cv = kfold_cv(model, X, y, n_splits=10, random_state=42)
        mc = monte_carlo_cv(
            model, X, y, n_splits=100, test_size=0.20, random_state=42,
            fallback_predictions=cv.predictions,
        )
        cv_fold_metrics.append(cv.fold_metrics)
        mc_fold_metrics.append(mc.fold_metrics)

    cv_mean = fold_mean_table(cv_fold_metrics, MODEL_ORDER)
    sorted_cv_names = cv_mean["Model_ID"].tolist()
    cv_std = fold_std_table(
        [cv_fold_metrics[MODEL_ORDER.index(n)] for n in sorted_cv_names],
        sorted_cv_names,
    )

    mc_mean = fold_mean_table(mc_fold_metrics, MODEL_ORDER)
    sorted_mc_names = mc_mean["Model_ID"].tolist()
    mc_std = fold_std_table(
        [mc_fold_metrics[MODEL_ORDER.index(n)] for n in sorted_mc_names],
        sorted_mc_names,
    )

    cv_mean_sd = _fmt_mean_sd(cv_mean, cv_std)
    mc_mean_sd = _fmt_mean_sd(mc_mean, mc_std)

    with pd.ExcelWriter(OUT_PATH) as writer:
        cv_df.to_excel(writer, sheet_name="10Fold_CV", index=False)
        mc_df.to_excel(writer, sheet_name="MonteCarlo_CV", index=False)
        cv_mean_sd.to_excel(writer, sheet_name="10Fold_Mean±SD", index=False)
        mc_mean_sd.to_excel(writer, sheet_name="MonteCarlo_Mean±SD", index=False)
        params_df.to_excel(writer, sheet_name="Tuned_Hyperparameters", index=False)

    print(f"Saved: {OUT_PATH}\n")
    print("=== 10Fold_CV (official, pooled) ===")
    print(cv_df.to_string(index=False))
    print("\n=== MonteCarlo_CV (official, avg-of-repeat) ===")
    print(mc_df.to_string(index=False))
    print("\n=== 10Fold_Mean±SD (recomputed, mean-of-fold ± SD across 10 folds) ===")
    print(cv_mean_sd.to_string(index=False))
    print("\n=== MonteCarlo_Mean±SD (recomputed, mean ± SD across 100 repeats) ===")
    print(mc_mean_sd.to_string(index=False))


if __name__ == "__main__":
    main()
