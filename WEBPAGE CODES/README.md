# How the Geopolymer Mortar Strength Predictor webpage was built

Pipeline, run in order:

1. **`1_extract_final.py`** — loads the tuned XGBoost model (from the 200-trial
   Optuna search), extracts all 850 trees into a compact JSON format
   (`{"f": feature_index, "t": threshold, "l"/"r": child indices, "m": default
   direction}` per split node, `{"leaf": weight}` per leaf), and validates
   that reimplementing prediction from this compact JSON reproduces the
   real model's output exactly (both against float32-quantized training
   values and realistic float64 user input) before trusting it for the page.
   Writes `xgb_trees_final.json`.

   Note: this script's input path (a `tuning_cache` pickle from an earlier,
   since-deleted results folder) no longer exists on disk, so this specific
   script can't be rerun today. It's kept for provenance — its output
   (`2_xgb_trees_final.json`) is what actually matters downstream, and that
   file is unaffected.

2. **`2_xgb_trees_final.json`** — the extracted tree data itself (output of
   step 1, input to step 4). 850 trees, ~244KB.

3. **`3_predictor_template.html`** — the page shell: layout, styling, mix-form
   field rendering (including the editable/auto-calculated Sand field),
   volume-balance calculator, and the in-browser prediction function that
   walks the tree JSON — with three placeholder comments
   (`/* __FEATURES__ */`, `/* __TREE_DATA__ */`, `/* __FIXTURES__ */`) where
   the model data gets injected.

4. **`4_assemble.py`** — reads the template and the tree JSON, substitutes
   the placeholders, and writes the final standalone HTML page (no build
   step, no server — everything, including the model, runs in the browser).

5. **`5_geopolymer_predictor_CURRENT_LIVE.html`** — today's actual published
   page, live at
   https://soranmanguri.github.io/Geopolymer-mortar-strength-predictor/

## Verified: rerunning (4) reproduces (5) byte-for-byte

`python 4_assemble.py` writes `geopolymer_predictor.html` next to it in this
folder. That output was diffed against `5_geopolymer_predictor_CURRENT_LIVE.html`
via SHA-256 hash and confirmed **identical** — same R² (0.925), same RMSE
(4.35 / 5.00 MPa), same "Sand" naming and manual-override behavior, same
everything. (Two small technical fixes were needed to get an exact match:
`3_predictor_template.html` was rebuilt directly from today's live page
rather than hand-edited, and `4_assemble.py` writes with explicit `\n`
line endings so Python's default Windows CRLF translation doesn't change
the output bytes.)

So: anyone who runs `python 4_assemble.py` in this folder gets exactly
today's page, not an older version.
