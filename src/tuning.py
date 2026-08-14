"""
Hyperparameter tuning using Optuna (Bayesian optimisation — TPE sampler).

Unlike random search, Optuna's TPE sampler learns from every completed trial
and steers subsequent suggestions toward regions that have shown high R².
This means fewer wasted evaluations and better final hyperparameters for the
same compute budget.

"""

from __future__ import annotations

import os
import joblib
import optuna
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# Suppress per-trial log spam; we show our own progress
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Default trial budgets
# ---------------------------------------------------------------------------

DEFAULT_N_TRIALS: dict = {
    "Extra Trees":   200,
    "Random Forest": 100,
    "CatBoost":      100,
    "XGBoost":       100,
}


# ---------------------------------------------------------------------------
# Objective functions  (one per model)
# ---------------------------------------------------------------------------

def _objective_et(trial, X, y, kf, random_state: int) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_features":      trial.suggest_float("max_features", 0.1, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_depth":         trial.suggest_int("max_depth", 5, 50),
        "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
        "ccp_alpha":         trial.suggest_float("ccp_alpha", 0.0, 0.1),
    }
    model = ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **params)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=1)
    return float(scores.mean())


def _objective_rf(trial, X, y, kf, random_state: int) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_features":      trial.suggest_float("max_features", 0.1, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_depth":         trial.suggest_int("max_depth", 5, 50),
        "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
        "ccp_alpha":         trial.suggest_float("ccp_alpha", 0.0, 0.1),
    }
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=1)
    return float(scores.mean())


def _objective_catboost(trial, X, y, kf, random_state: int) -> float:
    params = {
        "iterations":       trial.suggest_int("iterations", 500, 3000, step=100),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth":            trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
        "random_strength":  trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
    model = CatBoostRegressor(
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        **params,
    )
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=1)
    return float(scores.mean())


def _objective_xgboost(trial, X, y, kf, random_state: int) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }
    model = XGBRegressor(random_state=random_state, n_jobs=-1, **params)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=1)
    return float(scores.mean())


_OBJECTIVES = {
    "Extra Trees":   _objective_et,
    "Random Forest": _objective_rf,
    "CatBoost":      _objective_catboost,
    "XGBoost":       _objective_xgboost,
}


# ---------------------------------------------------------------------------
# Best estimator factory
# ---------------------------------------------------------------------------

def _make_best_estimator(model_name: str, best_params: dict, random_state: int):
    if model_name == "Extra Trees":
        return ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **best_params)
    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **best_params)
    if model_name == "CatBoost":
        return CatBoostRegressor(
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            **best_params,
        )
    if model_name == "XGBoost":
        return XGBRegressor(random_state=random_state, n_jobs=-1, **best_params)
    raise ValueError(f"Unknown model_name: {model_name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_model(
    model_name: str,
    X,
    y,
    n_trials: int | None = None,
    n_splits: int = 10,
    random_state: int = 42,
    verbose: bool = True,
    cache_dir: str = ".",
    reuse_cache: bool | None = None,
) -> tuple:
    """
    Tune a model via Optuna Bayesian optimisation (TPE sampler).

    Parameters
    ----------
    model_name   : "Extra Trees", "Random Forest", "CatBoost", or "XGBoost"
    X, y         : features and target from Dataset.
    n_trials     : Optuna trials; defaults to DEFAULT_N_TRIALS[model_name].
    n_splits     : folds per repeat; used as RepeatedKFold(n_splits, n_repeats=3).
    random_state : reproducibility seed.
    verbose      : print a progress summary every 50 trials.
    cache_dir    : folder for the ``tuned_*_results.pkl`` cache. Keep this
                   dataset-specific — a cache written for one dataset is not
                   valid for another.
    reuse_cache  : if a cache file exists, whether to load it instead of
                   re-tuning. ``None`` (default) asks interactively, which is
                   the historical behaviour; pass ``True``/``False`` to answer
                   up front (needed for non-interactive runs).

    Returns
    -------
    (best_estimator, best_params, results_df)
        best_estimator is fitted on the full dataset.
        results_df has one row per trial, sorted best-first.
    """
    if model_name not in _OBJECTIVES:
        raise ValueError(f"model_name must be one of {list(_OBJECTIVES)}")

    if n_trials is None:
        n_trials = DEFAULT_N_TRIALS[model_name]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    objective_fn = _OBJECTIVES[model_name]

    # Progress callback — prints every 50 trials
    def _progress(study, trial):
        if verbose and (trial.number + 1) % 50 == 0:
            print(f"  Trial {trial.number + 1:>4}/{n_trials}  |  "
                  f"Best R² so far: {study.best_value:.4f}")

    sampler = optuna.samplers.TPESampler(seed=random_state)

    model_safe_name = model_name.replace(" ", "_").lower()
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, f"tuned_{model_safe_name}_results.pkl")

    if os.path.exists(save_path):
        if reuse_cache is None:
            while True:
                ans = input(f"Saved tuning results for '{model_name}' found. Do you want to load it instead of running tuning again? (y/n): ").strip().lower()
                if ans in ['y', 'n']:
                    break
                print("Please answer 'y' or 'n'.")
        else:
            ans = 'y' if reuse_cache else 'n'

        if ans == 'y':
            if verbose:
                print(f"Loading saved tuning results for {model_name} from {save_path}...")
            return joblib.load(save_path)

    study = optuna.create_study(direction="maximize", sampler=sampler)

    if verbose:
        print(f"  Running {n_trials} Optuna trials × {n_splits}-fold CV ...")

    study.optimize(
        lambda trial: objective_fn(trial, X, y, kf, random_state),
        n_trials=n_trials,
        callbacks=[_progress],
        n_jobs=1,
    )

    best_params = study.best_params
    best_model = _make_best_estimator(model_name, best_params, random_state)
    best_model.fit(X, y)

    results_df = (
        study.trials_dataframe()
        .rename(columns={"value": "mean_cv_r2", "number": "trial_id"})
        .sort_values("mean_cv_r2", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"Saving tuned results for {model_name} to {save_path}...")
    joblib.dump((best_model, best_params, results_df), save_path)

    return best_model, best_params, results_df
