"""
SHAP analysis for the 200-trial-tuned XGBoost model on the NS30 dataset, using
Good version's own tuning artifacts.

Reproduces the same methodology as run_shap_ns30.py, but swaps in the tuned
model:
  dataset : Good version\\Results_NewDataset_Full_NS30\\Optimization - NewDataset (clean, NS30).xlsx
  model   : Good version\\tuned200_summary.pkl -> tuned_models["XGBoost"]
            (fitted XGBRegressor from the 200-trial Optuna search, n_estimators=850,
            max_depth=8, learning_rate=0.211, ... — see tuned_params["XGBoost"])
  grouping: Slag + Fly ash + Metakaoline -> "Total Binders (Slag/FA/MK)"

Output goes to a fresh 'SHAP_rerun_Tuned200' folder so neither the original
SHAP results nor the baseline-model SHAP_rerun results are touched.

Usage (from an activated venv; can be run from anywhere, 'src' resolves via
this script's own directory):
    python run_shap_ns30_tuned200.py
"""

from __future__ import annotations

import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: works headless, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data import load_data

GOOD_VERSION = Path(__file__).parent
DATA_PATH = GOOD_VERSION / "Optimization - NewDataset (clean, NS30).xlsx"
SUMMARY_PATH = GOOD_VERSION / "tuned200_summary.pkl"
OUTPUT_DIR = GOOD_VERSION / "SHAP_XGBoost_Tuned"

BINDER_FEATURES = ["GGBFS (kg/m³)", "Fly ash (kg/m³)", "Metakaoline (kg/m³)"]
BINDER_GROUP_NAME = "Total Binders (GGBFS/FA/MK)"


def _importance_table(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
    df = df.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    total = df["Mean |SHAP|"].sum()
    df["Importance (%)"] = df["Mean |SHAP|"] / total * 100.0 if total > 0 else 0.0
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_data(file_path=str(DATA_PATH), extra_features=["NS size(nm)"])
    print(dataset.summary())

    summary = joblib.load(SUMMARY_PATH)
    model = summary["tuned_models"]["XGBoost"]
    print("\nTuned XGBoost params:", summary["tuned_params"]["XGBoost"])
    print("n_features_in_:", model.n_features_in_)

    X = dataset.X.copy()
    X.columns = [dataset.display_name_map[c] for c in X.columns]  # original display names

    explainer = shap.TreeExplainer(model)
    explanation = explainer(dataset.X)
    explanation.feature_names = list(X.columns)
    explanation.data = X.to_numpy()

    # ---- individual (ungrouped) feature results -----------------------
    imp_df = _importance_table(explanation.values, list(X.columns))
    imp_df.to_excel(OUTPUT_DIR / "XGBoost_SHAP_Importance.xlsx", index=False)
    print("\n=== SHAP importance (individual features) ===")
    print(imp_df.to_string(index=False))

    plt.figure()
    shap.plots.bar(explanation, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "XGBoost_SHAP_Bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "XGBoost_SHAP_Beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- grouped: Slag + Fly ash + Metakaoline -> Total Binders --------
    other_features = [c for c in X.columns if c not in BINDER_FEATURES]
    grouped_names = [BINDER_GROUP_NAME] + other_features

    binder_idx = [X.columns.get_loc(c) for c in BINDER_FEATURES]
    other_idx = [X.columns.get_loc(c) for c in other_features]

    grouped_shap = np.column_stack([
        explanation.values[:, binder_idx].sum(axis=1),
        explanation.values[:, other_idx],
    ])
    grouped_data = np.column_stack([
        X[BINDER_FEATURES].to_numpy().sum(axis=1),
        X[other_features].to_numpy(),
    ])

    grouped_explanation = shap.Explanation(
        values=grouped_shap,
        base_values=explanation.base_values,
        data=grouped_data,
        feature_names=grouped_names,
    )

    grouped_imp_df = _importance_table(grouped_shap, grouped_names)
    grouped_imp_df.to_excel(OUTPUT_DIR / "XGBoost_SHAP_Importance_Grouped.xlsx", index=False)
    print("\n=== SHAP importance (grouped binders) ===")
    print(grouped_imp_df.to_string(index=False))

    plt.figure()
    shap.plots.bar(grouped_explanation, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "XGBoost_SHAP_Bar_Grouped.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.beeswarm(grouped_explanation, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "XGBoost_SHAP_Beeswarm_Grouped.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved SHAP outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
