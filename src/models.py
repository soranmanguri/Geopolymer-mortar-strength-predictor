"""
Pre-configured machine learning estimators used in the study.

The CatBoost configuration mirrors the hyperparameters chosen during the
exploratory phase of the project; Extra Trees uses the same configuration
that produced the best results in the paper (R^2 = 0.855 under 10-fold CV).
"""

from __future__ import annotations

from typing import Dict, Union

from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


DEFAULT_RANDOM_STATE = 42


def get_catboost_model(random_state: int = DEFAULT_RANDOM_STATE) -> CatBoostRegressor:
    """CatBoost regressor used as the gradient-boosting baseline."""
    return CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=10,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def get_extra_trees_model(random_state: int = DEFAULT_RANDOM_STATE) -> ExtraTreesRegressor:
    """Extra Trees regressor (best performer in the paper)."""
    return ExtraTreesRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )


def get_pycaret_models_dict(random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, Union[str, object]]:
    """
    Mapping of display name -> PyCaret estimator id (or pre-built estimator).

    PyCaret resolves string ids ('lr', 'rf', 'et', 'xgboost') against its
    internal model registry; CatBoost is supplied as a custom estimator so
    we can control its hyperparameters explicitly.
    """
    return {
        "Linear Regression": "lr",
        "Random Forest": "rf",
        "Extra Trees": "et",
        "XGBoost": "xgboost",
        "CatBoost": get_catboost_model(random_state=random_state),
    }
