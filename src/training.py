"""
End-to-end training pipeline for the five candidate models.

This wraps PyCaret's ``setup`` / ``create_model`` flow together with the
validation routines from :mod:`src.validation`, so the notebook can run the
whole comparison with a single function call.

Both validation strategies compute R2, Pearson r, RMSE, MAE, IoA, Theta Mean
and Theta CoV once per fold/repeat, then report the MEAN and SD of each
metric across all folds/repeats — never a single metric computed on pooled
predictions. Saves a single ``Results_Summary.xlsx`` (mean + mean±SD sheets
for both 10-fold CV and Monte Carlo CV), plus one final model per algorithm
(fit on 100% of the data) pickled to a ``Final_Models`` subfolder. No data
is held back for a separate test split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from pycaret.regression import create_model, pull, setup
from sklearn.base import clone

from .data import Dataset
from .metrics import fold_mean_table, fold_std_table, metrics_table
from .models import get_pycaret_models_dict
from .validation import CVResult, MonteCarloResult, kfold_cv, monte_carlo_cv


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResults:
    """All results produced by :func:`run_full_pipeline`."""

    trained_models: Dict[str, object]
    final_models: Dict[str, object]
    cv_details: pd.DataFrame
    pooled_10fold: pd.DataFrame
    pooled_10fold_std: pd.DataFrame
    monte_carlo_avg: pd.DataFrame
    monte_carlo_std: pd.DataFrame
    predictions_10fold: Dict[str, np.ndarray]
    predictions_mc: Dict[str, np.ndarray]
    raw_cv: Dict[str, CVResult] = field(default_factory=dict)
    raw_mc: Dict[str, MonteCarloResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(
    dataset: Dataset,
    output_dir: str = ".",
    n_mc_repeats: int = 100,
    mc_test_size: float = 0.20,
    random_state: int = 42,
) -> PipelineResults:
    """
    Run the full multi-model training & validation pipeline.

    All models are trained and evaluated on the full dataset — no separate
    holdout split is carved out — using 10-fold CV and Monte Carlo CV as the
    two evaluation strategies. (PyCaret's own internal train/test split from
    ``setup()`` only supplies each model's hyperparameter template via
    ``create_model()``; the actual CV fits below clone that template and
    refit it on the full ``dataset.X``/``dataset.y``, so PyCaret's internal
    split has no effect on the reported metrics.) Saves a single
    ``Results_Summary.xlsx`` to ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)
    X, y = dataset.X, dataset.y

    # ----------------------------- PyCaret setup --------------------------
    setup(
        data=dataset.data,
        target=dataset.target,
        session_id=random_state,
        fold=10,
        fold_shuffle=True,
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        verbose=False,
        html=False,
    )

    models_to_train = get_pycaret_models_dict(random_state=random_state)

    # ----------------------------- Train each model -----------------------
    trained_models: Dict[str, object] = {}
    cv_results_all: List[pd.DataFrame] = []

    for model_name, estimator in models_to_train.items():
        print(f"\n{'='*30}\nTraining: {model_name}\n{'='*30}")
        model = create_model(estimator)
        cv_table = pull().copy()
        cv_table["Model_ID"] = model_name
        trained_models[model_name] = model
        cv_results_all.append(cv_table)

    cv_details = pd.concat(cv_results_all, ignore_index=True)

    # ----------------------------- 10-fold CV ------------------------------
    # Each metric is computed once per fold (10 values per model), then the
    # reported table is the MEAN across those 10 fold-level values — not a
    # single metric computed on all pooled out-of-fold predictions at once.
    raw_cv: Dict[str, CVResult] = {}
    predictions_10fold: Dict[str, np.ndarray] = {}
    cv_fold_metrics_list = []

    for model_name, model in trained_models.items():
        cv = kfold_cv(model, X, y, n_splits=10, random_state=random_state)
        raw_cv[model_name] = cv
        predictions_10fold[model_name] = cv.predictions
        cv_fold_metrics_list.append(cv.fold_metrics)

    model_names = list(trained_models.keys())
    pooled_10fold = fold_mean_table(cv_fold_metrics_list, model_names)
    sorted_names_10fold = pooled_10fold["Model_ID"].tolist()
    cv_fold_metrics_sorted = [
        cv_fold_metrics_list[model_names.index(n)] for n in sorted_names_10fold
    ]
    pooled_10fold_std = fold_std_table(cv_fold_metrics_sorted, sorted_names_10fold)

    # ----------------------------- Monte Carlo ----------------------------
    raw_mc: Dict[str, MonteCarloResult] = {}
    predictions_mc: Dict[str, np.ndarray] = {}
    mc_rows = []

    for model_name, model in trained_models.items():
        mc = monte_carlo_cv(
            model, X, y,
            n_splits=n_mc_repeats,
            test_size=mc_test_size,
            random_state=random_state,
            fallback_predictions=predictions_10fold[model_name],
        )
        raw_mc[model_name] = mc
        predictions_mc[model_name] = mc.sample_predictions
        mc_rows.append(mc.avg_metrics)

    monte_carlo_avg = metrics_table(mc_rows, model_names)
    sorted_names_mc = monte_carlo_avg["Model_ID"].tolist()
    mc_fold_metrics_sorted = [
        raw_mc[n].fold_metrics for n in sorted_names_mc
    ]
    monte_carlo_std = fold_std_table(mc_fold_metrics_sorted, sorted_names_mc)

    # ----------------------------- Final models (100% of the data) --------
    # These are separate from the CV evaluation above: each one is fit on
    # every row, so it has no held-out data of its own to be scored against.
    # Its expected performance is what the CV metrics above already estimate.
    final_models: Dict[str, object] = {}
    models_dir = os.path.join(output_dir, "Final_Models")
    os.makedirs(models_dir, exist_ok=True)

    for model_name, model in trained_models.items():
        fm = clone(model)
        fm.fit(X, y)
        final_models[model_name] = fm
        safe_name = model_name.replace(" ", "_")
        joblib.dump(fm, os.path.join(models_dir, f"{safe_name}.pkl"))

    print(f"\nSaved final models (trained on 100% of the data) to: {models_dir}")

    # ----------------------------- Per-sample predictions ------------------
    # Out-of-fold predictions (each row predicted by a model that did not see
    # it during training for that fold/repeat) — NOT the final 100%-trained
    # models' in-sample predictions, which would be near-perfect/overfit and
    # not indicative of real predictive accuracy.
    predictions_df = pd.DataFrame({"Actual": y.reset_index(drop=True)})
    for model_name in model_names:
        col = model_name.replace(" ", "_")
        predictions_df[f"{col}_10Fold_Pred"] = predictions_10fold[model_name]
        predictions_df[f"{col}_MonteCarlo_Pred"] = predictions_mc[model_name]

    # ----------------------------- Save one summary Excel -----------------
    SHOW_METRICS = ["R2", "Pearson (r)", "RMSE", "MAE", "MAPE", "IoA", "Theta Mean", "Theta CoV"]

    def _fmt_mean_sd(mean_df: pd.DataFrame, sd_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, mrow in mean_df.iterrows():
            name = mrow["Model_ID"]
            srow = sd_df[sd_df["Model_ID"] == name].iloc[0]
            entry = {"Model": name}
            for m in SHOW_METRICS:
                entry[m] = f"{mrow[m]:.3f} ± {srow[m]:.3f}"
            rows.append(entry)
        return pd.DataFrame(rows)

    summary_path = os.path.join(output_dir, "Results_Summary.xlsx")
    with pd.ExcelWriter(summary_path) as writer:
        pooled_10fold.to_excel(writer, sheet_name="10Fold_CV", index=False)
        monte_carlo_avg.to_excel(writer, sheet_name="MonteCarlo_CV", index=False)
        _fmt_mean_sd(pooled_10fold, pooled_10fold_std).to_excel(writer, sheet_name="10Fold_Mean±SD", index=False)
        _fmt_mean_sd(monte_carlo_avg, monte_carlo_std).to_excel(writer, sheet_name="MonteCarlo_Mean±SD", index=False)
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
    print(f"\nSaved: {summary_path}")

    return PipelineResults(
        trained_models=trained_models,
        final_models=final_models,
        cv_details=cv_details,
        pooled_10fold=pooled_10fold,
        pooled_10fold_std=pooled_10fold_std,
        monte_carlo_avg=monte_carlo_avg,
        monte_carlo_std=monte_carlo_std,
        predictions_10fold=predictions_10fold,
        predictions_mc=predictions_mc,
        raw_cv=raw_cv,
        raw_mc=raw_mc,
    )
