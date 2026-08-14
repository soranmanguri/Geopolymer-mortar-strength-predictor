"""
One-time compatibility patch: shap==0.49.1's XGBTreeModelLoader parses
XGBoost's saved base_score as a plain float string, but xgboost>=3.0 writes
it bracketed, e.g. '[45.2]' instead of '45.2'. Without this patch,
run_shap_ns30_tuned200.py fails with:
    ValueError: could not convert string to float: '[...]'

This can't be fixed inside this project's own code (the bug is in a
constructor deep in shap's internals, not something callable/monkeypatchable
from outside), so this script edits the installed shap package directly --
the same fix, just made idempotent, reproducible and reviewable instead of
being a silent one-off edit to a random machine's site-packages.

Safe to run multiple times: it checks whether the patch is already applied
(or unnecessary, e.g. a future shap version fixes this upstream) before
changing anything.

Usage (from an activated venv, after `pip install -r requirements.txt`):
    python patch_shap_xgboost_compat.py
"""

from __future__ import annotations

import shap
from pathlib import Path

TARGET = Path(shap.__file__).parent / "explainers" / "_tree.py"

OLD = 'self.base_score = float(learner_model_param["base_score"])'
OLD2 = 'base_score = float(learner_model_param["base_score"])'
NEW = 'self.base_score = float(str(learner_model_param["base_score"]).strip("[]"))'
NEW2 = 'base_score = float(str(learner_model_param["base_score"]).strip("[]"))'


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    if NEW in text and NEW2 in text:
        print(f"Already patched: {TARGET}")
        return

    if OLD not in text or OLD2 not in text:
        print(
            f"Expected pattern not found in {TARGET}.\n"
            "Either this shap version already handles bracketed base_score "
            "natively (nothing to do), or its internals changed -- if "
            "run_shap_ns30_tuned200.py fails with a base_score ValueError, "
            "inspect this file manually."
        )
        return

    patched = text.replace(OLD, NEW).replace(OLD2, NEW2)
    TARGET.write_text(patched, encoding="utf-8")
    print(f"Patched: {TARGET}")


if __name__ == "__main__":
    main()
