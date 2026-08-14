"""
The original 200-trial Optuna tuning run (Extra Trees, Random Forest,
CatBoost, XGBoost) that produced tuned200_summary.pkl.

This is a straight port of the script actually executed on 2026-08-07 (it
ran for ~3.25 hours: Extra Trees 17.6 min, Random Forest 24.5 min, CatBoost
31.3 min, XGBoost 121.4 min). Paths have been updated to be relative to this
folder (the original script pointed at "Results_NewDataset_Full" /
"Results_NewDataset_Full_Tuned200", which have since been deleted) and to
reuse the dataset copy already restored here
("Optimization - NewDataset (clean, all rows).xlsx"). The tuning logic
itself (Optuna search spaces, CV setup) is unchanged -- see src/tuning.py.

Re-running this from scratch takes hours and will OVERWRITE
tuned200_summary.pkl. There should be no need to rerun it -- it's kept here
for reproducibility/audit purposes, not as a step in the normal pipeline.

Usage (from an activated venv, run from the Good version directory so 'src'
resolves):
    python run_tuning_200.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib

from src.data import load_data
from src.validation import kfold_cv, monte_carlo_cv
from src.tuning import tune_model

HERE = Path(__file__).parent
CLEAN_FILE = HERE / "Optimization - NewDataset (clean, all rows).xlsx"
RESULTS_DIR = HERE
TUNING_CACHE = RESULTS_DIR / "tuning_cache"
SUMMARY_PATH = RESULTS_DIR / "tuned200_summary.pkl"
EXTRA_FEATURES = ["NS size(nm)"]
RANDOM_STATE = 42
N_TRIALS = 200
N_MC_REPEATS = 100

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TUNING_CACHE, exist_ok=True)

ds = load_data(str(CLEAN_FILE), extra_features=EXTRA_FEATURES)
print(ds.summary())

MODELS_TO_TUNE = ["Extra Trees", "Random Forest", "CatBoost", "XGBoost"]

tuned_models = {}
tuned_params = {}
tuned_cv_r2 = {}
tuned_mc_r2 = {}

for name in MODELS_TO_TUNE:
    print(f"\n{'='*60}\nTuning {name} with {N_TRIALS} Optuna trials\n{'='*60}")
    t0 = time.time()
    best_model, best_params, results_df = tune_model(
        name, ds.X, ds.y,
        n_trials=N_TRIALS, n_splits=10, random_state=RANDOM_STATE,
        verbose=True, cache_dir=str(TUNING_CACHE), reuse_cache=False,
    )
    elapsed = time.time() - t0
    print(f"{name} tuning done in {elapsed/60:.1f} min")
    print(f"Best params: {best_params}")

    cv = kfold_cv(best_model, ds.X, ds.y, n_splits=10, random_state=RANDOM_STATE)
    mc = monte_carlo_cv(
        best_model, ds.X, ds.y,
        n_splits=N_MC_REPEATS, test_size=0.20, random_state=RANDOM_STATE,
        fallback_predictions=cv.predictions,
    )
    tuned_models[name] = best_model
    tuned_params[name] = best_params
    tuned_cv_r2[name] = cv.metrics
    tuned_mc_r2[name] = mc.avg_metrics
    print(f"{name} TUNED  10-fold R2 (pooled): {cv.metrics['R2']:.4f}  |  MC R2: {mc.avg_metrics['R2']:.4f}")

print("\n\n" + "=" * 60)
print(f"FINAL SUMMARY - Optuna {N_TRIALS} trials, dataset: 136 rows (median NS fill)")
print("=" * 60)
print(f"{'Model':<15} {'10F R2 (pooled)':>16} {'MC R2':>10}")
for name in MODELS_TO_TUNE:
    print(f"{name:<15} {tuned_cv_r2[name]['R2']:>16.4f} {tuned_mc_r2[name]['R2']:>10.4f}")

joblib.dump(
    {"tuned_models": tuned_models, "tuned_params": tuned_params,
     "tuned_cv": tuned_cv_r2, "tuned_mc": tuned_mc_r2},
    SUMMARY_PATH,
)
print(f"\nSaved summary to {SUMMARY_PATH}")
