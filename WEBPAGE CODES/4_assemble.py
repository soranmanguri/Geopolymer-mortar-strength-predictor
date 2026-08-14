# -*- coding: utf-8 -*-
"""
Assembles the final geopolymer_predictor.html by injecting the extracted
XGBoost tree data (2_xgb_trees_final.json) and this script's own FEATURES/
fixtures metadata into 3_predictor_template.html's placeholders.

Rerunning this reproduces the current live page byte-for-byte -- verified
directly against the actual published HTML (see README.md).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

with open(HERE / "2_xgb_trees_final.json", "r", encoding="utf-8") as f:
    trees_json_text = f.read()  # already compact JSON array text

BASE_SCORE = 44.91206

FEATURES = [
    {"key": "Molarity_M", "label": "NaOH Molarity", "unit": "M", "group": "activator",
     "min": 3, "max": 16, "step": 0.5, "def": 10},
    {"key": "Fly_ash", "label": "Fly Ash", "unit": "kg/m³", "group": "binder",
     "min": 0, "max": 734, "step": 1, "def": 343},
    {"key": "Slag", "label": "GGBFS Slag", "unit": "kg/m³", "group": "binder",
     "min": 0, "max": 700, "step": 1, "def": 341},
    {"key": "Metakaoline", "label": "Metakaolin", "unit": "kg/m³", "group": "binder",
     "min": 0, "max": 450, "step": 1, "def": 0},
    {"key": "Fine_Aggregate", "label": "Sand", "unit": "kg/m³", "group": "physical",
     "min": 432, "max": 1500, "step": 1, "def": 1162},
    {"key": "Nano_Silica_Kg", "label": "Nano-Silica Dosage", "unit": "kg/m³", "group": "nanosilica",
     "min": 0, "max": 50, "step": 0.5, "def": 13},
    {"key": "NaOH", "label": "NaOH Solution", "unit": "kg/m³", "group": "activator",
     "min": 64.28, "max": 175, "step": 0.5, "def": 108},
    {"key": "Na2SiO3", "label": "Na₂SiO₃ Solution", "unit": "kg/m³", "group": "activator",
     "min": 127.27, "max": 250, "step": 0.5, "def": 175},
    {"key": "Na2SiO3_NaOH", "label": "Na₂SiO₃ / NaOH Ratio", "unit": "", "group": "activator",
     "min": 1.0, "max": 2.5, "step": 0.05, "def": 1.75},
    {"key": "Alk_Binder", "label": "Alkaline / Binder Ratio", "unit": "", "group": "physical",
     "min": 0.3, "max": 0.7, "step": 0.01, "def": 0.5},
    {"key": "Curing_Condition", "label": "Curing Temperature", "unit": "°C", "group": "physical",
     "min": 20, "max": 70, "step": 1, "def": 25,
     "suggest": [20, 23, 25, 27, 28, 60, 70]},
    {"key": "NS_size_nm", "label": "Nano-Silica Particle Size", "unit": "nm", "group": "nanosilica",
     "min": 0, "max": 30, "step": 0.5, "def": 18.5,
     "suggest": [0, 10, 15, 17, 20, 25, 30]},
]

fixtures_raw = [
    {"row": [8.0, 500.0, 0.0, 0.0, 1500.0, 0.0, 72.73, 127.27, 1.75, 0.4, 60.0, 0.0], "expected": 18.1923},
    {"row": [10.0, 92.12, 368.49, 0.0, 1463.0, 6.9, 78.96, 197.41, 2.5, 0.6, 60.0, 15.0], "expected": 82.7377},
    {"row": [8.0, 339.5, 339.5, 0.0, 1122.0, 21.0, 175.0, 175.0, 1.0, 0.5, 25.0, 20.0], "expected": 45.8842},
    {"row": [12.0, 350.0, 350.0, 0.0, 1033.5, 0.0, 100.0, 250.0, 2.5, 0.5, 23.0, 0.0], "expected": 68.3023},
    {"row": [10.0, 130.0, 520.0, 0.0, 1300.0, 13.0, 65.0, 163.0, 2.5, 0.35, 60.0, 15.0], "expected": 59.0801},
]

features_js = "var FEATURES = " + json.dumps(FEATURES, ensure_ascii=False) + ";"
trees_js = "var BASE_SCORE = " + repr(BASE_SCORE) + ";\nvar TREES = " + trees_json_text + ";"
fixtures_js = "var FIXTURES = " + json.dumps(fixtures_raw, ensure_ascii=False) + ";"

with open(HERE / "3_predictor_template.html", "r", encoding="utf-8") as f:
    template = f.read()

template = template.replace("/* __FEATURES__ */", features_js)
template = template.replace("/* __TREE_DATA__ */", trees_js)
template = template.replace("/* __FIXTURES__ */", fixtures_js)

out_path = HERE / "geopolymer_predictor.html"
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(template)

print("Written:", out_path)
print("Size (KB):", len(template.encode("utf-8")) / 1024)
