"""
Pure computation logic for the Geopolymer Mortar Strength Predictor,
factored out of app.py so it can be unit-tested without a Streamlit
runtime. app.py imports these functions for its widget callbacks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

FEATURES = [
    {"key": "Molarity_M", "label": "NaOH Molarity", "min": 3.0, "max": 16.0},
    {"key": "Fly_ash", "label": "Fly Ash", "min": 0.0, "max": 734.0},
    {"key": "Slag", "label": "GGBFS Slag", "min": 0.0, "max": 700.0},
    {"key": "Metakaoline", "label": "Metakaolin", "min": 0.0, "max": 450.0},
    {"key": "Fine_Aggregate", "label": "Sand", "min": 432.0, "max": 1500.0},
    {"key": "Nano_Silica_Kg", "label": "Nano-Silica Dosage", "min": 0.0, "max": 50.0},
    {"key": "NaOH", "label": "NaOH Solution", "min": 64.28, "max": 175.0},
    {"key": "Na2SiO3", "label": "Na2SiO3 Solution", "min": 127.27, "max": 250.0},
    {"key": "Na2SiO3_NaOH", "label": "Na2SiO3 / NaOH Ratio", "min": 1.0, "max": 2.5},
    {"key": "Alk_Binder", "label": "Alkaline / Binder Ratio", "min": 0.3, "max": 0.7},
    {"key": "Curing_Condition", "label": "Curing Temperature", "min": 20.0, "max": 70.0},
    {"key": "NS_size_nm", "label": "Nano-Silica Particle Size", "min": 0.0, "max": 30.0},
]
FEATURE_KEYS = [f["key"] for f in FEATURES]

SG_DEFAULTS = {
    "Fly_ash": 2.2, "Slag": 2.85, "Metakaoline": 2.6, "Fine_Aggregate": 2.65,
    "NaOH": 1.4, "Na2SiO3": 1.45, "Nano_Silica_Kg": 2.2,
}
VOID_PCT_DEFAULT = 1.5
VOLUME_TARGET_L = 1000.0
BINDER_KEYS = ["Fly_ash", "Slag", "Metakaoline"]


def compute_na2sio3_naoh_ratio(naoh: float, na2sio3: float) -> float:
    return na2sio3 / naoh if naoh > 0 else 0.0


def compute_alk_binder_ratio(naoh: float, na2sio3: float, fly_ash: float, slag: float, metakaolin: float) -> float:
    binder = fly_ash + slag + metakaolin
    return (naoh + na2sio3) / binder if binder > 0 else 0.0


def compute_sand_auto(values: Dict[str, float], sg: Dict[str, float] | None = None, void_pct: float = VOID_PCT_DEFAULT) -> float:
    """Volume-balance auto-fill: same absolute-volume method as the JS version."""
    sg = {**SG_DEFAULTS, **(sg or {})}
    occupied = 0.0
    for key in ["Fly_ash", "Slag", "Metakaoline", "NaOH", "Na2SiO3", "Nano_Silica_Kg"]:
        mass = values.get(key, 0.0)
        s = sg.get(key, 1.0)
        if s > 0:
            occupied += mass / s
    occupied += (void_pct / 100.0) * VOLUME_TARGET_L
    remaining = VOLUME_TARGET_L - occupied
    fine_agg_vol = max(0.0, remaining)
    return fine_agg_vol * sg["Fine_Aggregate"]


def compute_violations(vector: List[float]) -> List[str]:
    """vector must be in FEATURE_KEYS order."""
    violations = []
    for f, v in zip(FEATURES, vector):
        if v < f["min"] or v > f["max"]:
            violations.append(f"{f['label']}={v:g} (allowed {f['min']:g}–{f['max']:g})")
    binder_idx = [FEATURE_KEYS.index(k) for k in BINDER_KEYS]
    total_binder = sum(vector[i] for i in binder_idx)
    if total_binder <= 0:
        violations.append("Total Binder (Fly Ash + GGBFS Slag + Metakaolin) = 0 kg/m³ (at least one binder is required)")
    return violations


def build_vector(values: Dict[str, float]) -> List[float]:
    v = dict(values)
    v["Na2SiO3_NaOH"] = compute_na2sio3_naoh_ratio(v.get("NaOH", 0), v.get("Na2SiO3", 0))
    v["Alk_Binder"] = compute_alk_binder_ratio(v.get("NaOH", 0), v.get("Na2SiO3", 0), v.get("Fly_ash", 0), v.get("Slag", 0), v.get("Metakaoline", 0))
    return [v.get(k, 0.0) for k in FEATURE_KEYS]


def predict(model, vector: List[float]) -> float:
    return float(model.predict(np.array([vector], dtype=np.float32))[0])
