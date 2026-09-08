import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
from predictor_logic import (
    compute_na2sio3_naoh_ratio, compute_alk_binder_ratio, compute_sand_auto,
    compute_violations, build_vector, predict, FEATURE_KEYS,
)

model = joblib.load(Path(__file__).parent / "xgboost_tuned_model.pkl")

print("=== TEST 1: default mix ratios ===")
r = compute_na2sio3_naoh_ratio(108, 175)
print(f"Na2SiO3/NaOH (expect 1.620): {r:.3f}")
ab = compute_alk_binder_ratio(108, 175, 343, 341, 0)
print(f"Alk:Binder (expect 0.414): {ab:.3f}")

print("\n=== TEST 2: Sand auto-calc, default mix (expect ~1340.1, matches JS) ===")
defaults = {"Fly_ash": 343, "Slag": 341, "Metakaoline": 0, "NaOH": 108, "Na2SiO3": 175, "Nano_Silica_Kg": 13}
sand = compute_sand_auto(defaults)
print(f"Sand: {sand:.1f}")

print("\n=== TEST 3: full prediction, default mix (expect 62.8, matches JS initial load) ===")
values = {**defaults, "Molarity_M": 10, "Fine_Aggregate": sand, "Curing_Condition": 25, "NS_size_nm": 18.5}
vector = build_vector(values)
pred = predict(model, vector)
print(f"Predicted: {pred:.1f} MPa")
print(f"Vector: {dict(zip(FEATURE_KEYS, vector))}")
print(f"Violations: {compute_violations(vector)}")

print("\n=== TEST 4: all binders = 0 (expect blocked) ===")
values2 = {**values, "Fly_ash": 0, "Slag": 0, "Metakaoline": 0}
sand2 = compute_sand_auto(values2)
values2["Fine_Aggregate"] = sand2
vector2 = build_vector(values2)
violations2 = compute_violations(vector2)
print(f"Sand (expect ~2070.3): {sand2:.1f}")
print(f"Violations: {violations2}")

print("\n=== TEST 5: Sand=0 manually (expect blocked, mortar-validity) ===")
values3 = {**values, "Fine_Aggregate": 0}
vector3 = build_vector(values3)
violations3 = compute_violations(vector3)
print(f"Violations: {violations3}")

print("\n=== TEST 6: Fly Ash = 9999 (expect blocked, out of range) ===")
values4 = {**values, "Fly_ash": 9999}
vector4 = build_vector(values4)
violations4 = compute_violations(vector4)
print(f"Violations: {violations4}")

print("\n=== TEST 7: cross-check against known JS/model fixtures ===")
fixtures = [
    {"row": [8.0, 500.0, 0.0, 0.0, 1500.0, 0.0, 72.73, 127.27, 1.75, 0.4, 60.0, 0.0], "expected": 18.1923},
    {"row": [10.0, 92.12, 368.49, 0.0, 1463.0, 6.9, 78.96, 197.41, 2.5, 0.6, 60.0, 15.0], "expected": 82.7377},
]
for fx in fixtures:
    p = predict(model, fx["row"])
    print(f"expected={fx['expected']:.4f}  got={p:.4f}  match={abs(p-fx['expected'])<0.001}")
