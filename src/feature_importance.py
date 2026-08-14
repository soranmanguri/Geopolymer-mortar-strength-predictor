"""
Feature importance utilities.

Tree-based models in scikit-learn expose ``feature_importances_``; CatBoost
exposes ``get_feature_importance()``. ``compute_feature_importance`` hides
that detail and also normalizes the values to percentages.

The categorical grouping (Total Binders, Nano Silica, NaOH ...) follows the
discussion in the paper, where slag, fly ash and metakaolin are aggregated
because they jointly play the role of "precursor binder".
"""

from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np
import pandas as pd


def _raw_importances(model) -> np.ndarray:
    """Pull importances from either an sklearn or a CatBoost estimator."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "get_feature_importance"):
        return np.asarray(model.get_feature_importance(), dtype=float)
    raise AttributeError(
        f"Model of type {type(model).__name__} does not expose feature importances."
    )


def compute_feature_importance(
    model,
    feature_names: List[str],
    display_name_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a tidy feature-importance DataFrame.

    Returns a DataFrame with columns:
    ``Feature``, ``Importance``, ``Importance (%)``, ``Feature_Display``,
    sorted from most to least important.
    """
    importances = _raw_importances(model)
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Model returned {len(importances)} importances but {len(feature_names)} "
            "feature names were provided."
        )

    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)

    total = df["Importance"].sum()
    df["Importance (%)"] = df["Importance"] / total * 100.0 if total > 0 else 0.0

    if display_name_map is not None:
        df["Feature_Display"] = df["Feature"].map(lambda f: display_name_map.get(f, f))
    else:
        df["Feature_Display"] = df["Feature"]

    return df


def categorize_importance(fi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group binders together and return a categorical importance DataFrame.

    Lookups use the original (display) names so the function works regardless
    of whether the underlying feature names are 'Fly_ash' or 'Fly ash'.
    Importances that cannot be resolved fall back to 0.
    """
    imp_dict: Dict[str, float] = dict(
        zip(fi_df["Feature_Display"], fi_df["Importance (%)"])
    )

    categorical_data = {
        "Total Binders (Slag/FA/MK)": (
            imp_dict.get("Slag", 0)
            + imp_dict.get("Fly ash", 0)
            + imp_dict.get("Metakaoline", 0)
        ),
        "Nano Silica": imp_dict.get("Nano Silica (Kg)", 0),
        "NaOH Content": imp_dict.get("NaOH", 0),
        "Molarity": imp_dict.get("Molarity (M)", 0),
        "Curing Condition": imp_dict.get("Curing Condition", 0),
        "Fine Aggregate": imp_dict.get("Fine Aggregate", 0),
        "Alk: Binder Ratio": imp_dict.get("Alk: Binder", 0),
        "Na2SiO3 Content": imp_dict.get("Na2SiO3", 0),
        "Na2SiO3/NaOH Ratio": imp_dict.get("Na2SiO3/NaOH", 0),
        "NS Particle Size": imp_dict.get("NS size(nm)", 0),
    }

    return (
        pd.DataFrame(list(categorical_data.items()), columns=["Category", "Importance (%)"])
        .sort_values("Importance (%)", ascending=False)
        .reset_index(drop=True)
    )


def binder_breakdown(fi_df: pd.DataFrame) -> pd.DataFrame:
    """Return only the three binder rows (Slag, Fly ash, Metakaoline)."""
    imp_dict: Dict[str, float] = dict(
        zip(fi_df["Feature_Display"], fi_df["Importance (%)"])
    )
    rows = [
        ("Slag", imp_dict.get("Slag", 0)),
        ("Fly ash", imp_dict.get("Fly ash", 0)),
        ("Metakaoline", imp_dict.get("Metakaoline", 0)),
    ]
    return pd.DataFrame(rows, columns=["Binder", "Importance (%)"])
