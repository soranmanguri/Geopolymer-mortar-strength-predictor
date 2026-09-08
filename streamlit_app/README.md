# Streamlit version of the predictor

A Python/Streamlit reimplementation of the calculator at
https://soranmanguri.github.io/Geopolymer-mortar-strength-predictor/ —
same tuned XGBoost model (850 trees, 200-trial Optuna search, R²=0.925
10-fold CV, 136-mix integrated dataset), same input ranges, same Sand
auto-calculation via volume balance (with manual override), same live
Na₂SiO₃/NaOH and Alk:Binder ratios, and the same validation rules (every
input must stay within its trained range, and at least one binder must be
nonzero).

The static HTML/JS version runs entirely in the browser with the model's
tree structure embedded as JSON; this version runs the actual trained
`XGBRegressor` object server-side via Streamlit.

## Files

- **`app.py`** — the Streamlit UI (widgets, layout)
- **`predictor_logic.py`** — the actual calculations (ratios, Sand
  auto-calc, validation, prediction), factored out so they're unit-testable
  without a Streamlit runtime
- **`test_predictor_logic.py`** — verifies `predictor_logic.py` against
  known-correct values (matches the JS version's own test fixtures and
  initial-load result), run before every deploy
- **`xgboost_tuned_model.pkl`** — the tuned XGBoost model, exported from
  `tuned200_summary.pkl` (verified to reproduce its predictions exactly —
  see `test_predictor_logic.py`, TEST 7)
- **`requirements.txt`** — pinned dependencies

## Running locally

```
pip install -r requirements.txt
python test_predictor_logic.py   # verify correctness first
streamlit run app.py
```

## Deploying

Deployed via Streamlit Community Cloud, pointed at this repo/branch with
main file path `streamlit_app/app.py`.
