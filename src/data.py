"""
Data loading and preprocessing for the geopolymer mortar dataset.

The raw Excel file uses column names with spaces, parentheses and special
characters (e.g. 'Comp.(MPa, 28 days)', 'Alk: Binder'). PyCaret and several
scikit-learn pipelines do not tolerate those characters, so each column is
re-mapped to a "safe" snake_case identifier. Both the original and safe names
are kept so plots and tables can show human-readable labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Constants describing the dataset schema
# ---------------------------------------------------------------------------

DEFAULT_FILE_PATH = "Optimization - Final.xlsx"
DEFAULT_SHEET_NAME = "Sheet1"

FEATURE_NAMES_ORIGINAL: List[str] = [
    "Molarity (M)",
    "Fly ash (kg/m³)",
    "GGBFS (kg/m³)",
    "Metakaoline (kg/m³)",
    "Sand (kg/m³)",
    "Nano Silica (kg/m³)",
    "NaOH (kg/m³)",
    "Na₂SiO₃ (kg/m³)",
    "Na₂SiO₃/NaOH",
    "Alk: Binder",
    "Curing Condition (°C)",
]

TARGET_ORIGINAL = "Comp.(MPa, 28 days)"

# Binder columns whose missing values must be filled with 0
# (a missing value here means "this binder was not used in the mix").
BINDER_COLUMNS: List[str] = ["Fly ash (kg/m³)", "GGBFS (kg/m³)", "Metakaoline (kg/m³)"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_safe(name: str) -> str:
    """Convert any column name into a safe snake_case identifier."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name))
    name = re.sub(r"_+", "_", name).strip("_")
    return name


@dataclass
class Dataset:
    """Container for the prepared dataset and its metadata."""

    data: pd.DataFrame              # Full dataframe (features + target, safe names)
    X: pd.DataFrame                 # Feature matrix (safe names)
    y: pd.Series                    # Target vector
    feature_names: List[str]        # Safe feature names, in canonical order
    target: str                     # Safe target name
    rename_map: Dict[str, str]      # original -> safe
    display_name_map: Dict[str, str]  # safe -> original (for plot labels)

    def summary(self) -> str:
        return (
            f"Dataset shape: {self.data.shape}\n"
            f"Features ({len(self.feature_names)}): {self.feature_names}\n"
            f"Target: {self.target}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_data(
    file_path: str = DEFAULT_FILE_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
    extra_features: List[str] | None = None,
) -> Dataset:
    """
    Load the geopolymer mortar dataset from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the experimental dataset.
    sheet_name : str
        Name of the sheet to read.
    extra_features : list of str, optional
        Additional original-name columns to append to the feature set, e.g.
        ``["NS size(nm)"]`` for datasets that record nano-silica particle size.
        Omit it and the historical 11-feature schema is used unchanged.

    Returns
    -------
    Dataset
        Prepared dataset wrapped together with its metadata.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Strip hidden whitespace from column headers
    df.columns = df.columns.str.strip()

    # Fill missing binder values with 0 (binder simply not used in that mix)
    for col in BINDER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Curing Condition is a temperature in degrees C; a handful of entries carry
    # a literal '°C' suffix (e.g. '25°C') instead of a plain number.
    if "Curing Condition (°C)" in df.columns:
        df["Curing Condition (°C)"] = (
            df["Curing Condition (°C)"].astype(str).str.extract(r"([\d.]+)").astype(float)
        )

    # Build a stable original -> safe rename map for ALL columns
    rename_map = {col: make_safe(col) for col in df.columns}
    df = df.rename(columns=rename_map)

    feature_cols = list(FEATURE_NAMES_ORIGINAL) + list(extra_features or [])
    missing = [c for c in feature_cols if c not in rename_map]
    if missing:
        raise KeyError(f"Columns not found in {file_path}: {missing}")

    feature_names = [rename_map[c] for c in feature_cols]
    target = rename_map[TARGET_ORIGINAL]

    # Drop rows with missing target
    df = df.dropna(subset=[target]).copy()

    data = df[feature_names + [target]].copy()
    X = data[feature_names].copy()
    y = data[target].copy()

    # Build the inverse map for plotting (safe -> original)
    display_name_map = {v: k for k, v in rename_map.items()}

    return Dataset(
        data=data,
        X=X,
        y=y,
        feature_names=feature_names,
        target=target,
        rename_map=rename_map,
        display_name_map=display_name_map,
    )
