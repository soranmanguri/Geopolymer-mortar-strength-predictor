"""
Runner script: loads the geopolymer mortar dataset and runs the full
training/validation pipeline (CatBoost, Extra Trees, Random Forest,
XGBoost, Linear Regression) via src/training.py.

Usage (from an activated venv):
    python run_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

from src.data import load_data
from src.training import run_full_pipeline

HERE = Path(__file__).parent
DATA_PATH = HERE / "Optimization - NewDataset (clean, NS30).xlsx"
OUTPUT_DIR = HERE / "Results_Baseline"


def main() -> None:
    dataset = load_data(file_path=str(DATA_PATH), extra_features=["NS size(nm)"])
    print(dataset.summary())

    results = run_full_pipeline(dataset, output_dir=OUTPUT_DIR)

    print("\n=== 10-Fold CV (pooled) ===")
    print(results.pooled_10fold.to_string(index=False))

    print("\n=== Monte Carlo (100 repeats, avg) ===")
    print(results.monte_carlo_avg.to_string(index=False))


if __name__ == "__main__":
    main()
