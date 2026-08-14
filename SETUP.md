# Setup / reproduction

## 1. Environment

Tested with **Python 3.11.9** on Windows.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python patch_shap_xgboost_compat.py
```

The patch step is required once per environment — see the docstring in
`patch_shap_xgboost_compat.py` for why (a real incompatibility between
`shap==0.49.1` and `xgboost>=3.0`'s bracketed `base_score` format, not
something fixable from this project's own code).

`pycaret` (used only by `run_pipeline.py`, for the baseline models) pulls in
a large dependency tree of its own. Skip installing it if you only need the
tuned-model scripts (`run_tuned_*.py`, `run_shap_ns30_tuned200.py`).

## 2. Data

- `Optimization - NewDataset (clean, all rows).xlsx` — the exact dataset the
  200-trial tuning run (`run_tuning_200.py`) used. Missing `NS size(nm)`
  values are filled with the column median (20nm).
- `Optimization - NewDataset (clean, NS30).xlsx` — a later dataset version
  (missing `NS size(nm)` filled with a fixed 30nm instead) used by the
  baseline pipeline, SHAP, and feature-importance scripts.

These differ on only 4 of 136 rows but are **not interchangeable** — using
the wrong one silently shifts reported R² (see `run_tuned_plots.py`'s
docstring for a worked example of exactly this happening).

## 3. Reproducing results, in order

| Script | What it does | Runtime |
|---|---|---|
| `run_pipeline.py` | Baseline (untuned) models: LR, RF, ET, XGBoost, CatBoost | ~1 min |
| `run_tuning_200.py` | The 200-trial Optuna search that produced `tuned200_summary.pkl` | **~3.25 hours** — already done; rerunning overwrites it |
| `run_tuned_summary.py` | `Results_Summary_Tuned200.xlsx` from the stored tuning-time numbers | seconds |
| `run_tuned_plots.py` | `Scatter_TunedModels.png`, `Trend_TunedModels.png` | ~1 min |
| `run_tuned_bar_chart.py` | `Bar_TunedModels_R2.png` | seconds |
| `run_shap_ns30_tuned200.py` | SHAP importance plots for tuned XGBoost | ~1 min |

`run_tuning_200.py` does not need to be rerun to reproduce anything else —
every downstream script reads its output (`tuned200_summary.pkl`), which is
already included.

## 4. Reproducibility guarantee

With the exact pinned versions in `requirements.txt`, all CV metrics are
**bit-exact reproducible**: this was independently verified — a from-scratch
10-fold CV recompute on the correct dataset reproduced the officially stored
XGBoost R² (0.925003910044727) to all 15 decimal places. Different versions
of scikit-learn/xgboost/catboost/optuna can legitimately produce different
tree splits for the same `random_state`, so version drift — not randomness —
is the main risk to exact reproduction.
