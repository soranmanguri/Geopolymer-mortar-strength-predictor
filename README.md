# Geopolymer Mortar Strength Predictor

Machine learning pipeline predicting the 28-day compressive strength of
nano-silica-modified fly ash/slag/metakaolin geopolymer mortar from
mix-design parameters, plus a browser-based calculator implementing the
final tuned model.

**Live calculator:** https://soranmanguri.github.io/Geopolymer-mortar-strength-predictor/
(runs entirely in your browser — no installation, no server)

## Results

Four candidate models were tuned via a 200-trial Optuna (Bayesian TPE)
search over 10-fold cross-validation R². Official tuning-time metrics
(see `Results_Summary_Tuned200.xlsx` for the full table, including
standard deviation across folds/repeats):

| Model | 10-Fold CV R² | Monte Carlo R² (100 repeats) |
|---|---|---|
| **XGBoost** | **0.925** | 0.887 |
| CatBoost | 0.917 | 0.876 |
| Extra Trees | 0.903 | 0.863 |
| Random Forest | 0.875 | 0.845 |

XGBoost was selected as the final model (850 trees, tuned hyperparameters
in `Tuned_Hyperparameters` sheet of the same file) and is the model
embedded in the live calculator above.

## Repository structure

- **`src/`** — data loading, model tuning (Optuna search spaces), cross-
  validation, metrics, plotting, feature importance
- **`run_*.py`** — top-level scripts that reproduce each result (baseline
  models, the 200-trial tuning run, summary tables, plots, SHAP analysis)
- **`tuned200_summary.pkl`** — the fitted tuned models plus their CV metrics
- **`Results_Baseline/`, `Results_Summary_Tuned200.xlsx`** — baseline and
  tuned model comparison tables
- **`SHAP_XGBoost_Tuned/`** — SHAP feature-importance analysis for the
  final model
- **`WEBPAGE CODES/`** — the pipeline that extracts the tuned XGBoost model
  into the standalone HTML calculator (see its own README for details)
- **`docs/`** — the deployed copy of the calculator (GitHub Pages source)

## Reproducing these results

See **[`SETUP.md`](SETUP.md)** for environment setup (exact package
versions, a one-time compatibility patch needed for SHAP) and a full list
of what each script produces and how long it takes to run.

Reproducibility was independently verified: cross-validation metrics
recomputed from scratch in a clean environment reproduce the values above
to 15 decimal places, and the web calculator's generator pipeline
(`WEBPAGE CODES/4_assemble.py`) was confirmed to output the live page
byte-for-byte.

## Dataset

136 experimental mixes, 12 mix-design features (molarity, binder content,
nano-silica dosage and particle size, activator solution masses, curing
temperature, etc.), target: 28-day compressive strength (MPa). Two dataset
versions are included — see `SETUP.md` for which script uses which and why
they aren't interchangeable.
