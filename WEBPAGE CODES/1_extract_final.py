import sys, json
sys.path.insert(0, r"C:\Users\AMIN4PC\Desktop")

import numpy as np
import joblib
from src.data import load_data

best_model, best_params, results_df = joblib.load(
    r"C:\Users\AMIN4PC\Desktop\Results_NewDataset_Full_Tuned200\tuning_cache\tuned_xgboost_results.pkl"
)
booster = best_model.get_booster()
feature_names = list(best_model.feature_names_in_)

model_path = r"C:\Users\AMIN4PC\AppData\Local\Temp\claude\c--Users-AMIN4PC-mine\c7c3f6bc-bcf3-4a0b-b0cf-369e57334e3a\scratchpad\xgb_model.json"
booster.save_model(model_path)
with open(model_path) as f:
    model_json = json.load(f)

learner = model_json["learner"]
base_score = float(learner["learner_model_param"]["base_score"].strip("[]"))
trees_raw = learner["gradient_booster"]["model"]["trees"]
print("num trees:", len(trees_raw))

compact_trees = []
for t in trees_raw:
    left = t["left_children"]
    right = t["right_children"]
    split_idx = t["split_indices"]
    split_cond = t["split_conditions"]
    base_w = t["base_weights"]
    default_left = t["default_left"]
    n_nodes = len(left)
    arr = []
    for i in range(n_nodes):
        if left[i] == -1:  # leaf
            arr.append({"leaf": float(base_w[i])})
        else:
            arr.append({
                "f": int(split_idx[i]),
                "t": float(split_cond[i]),
                "l": int(left[i]),
                "r": int(right[i]),
                "m": int(default_left[i]),
            })
    compact_trees.append(arr)

compact_json = json.dumps(compact_trees, separators=(",", ":"))
print("compact JSON size:", len(compact_json), "bytes (~%.3f MB)" % (len(compact_json) / 1e6))

with open(r"C:\Users\AMIN4PC\AppData\Local\Temp\claude\c--Users-AMIN4PC-mine\c7c3f6bc-bcf3-4a0b-b0cf-369e57334e3a\scratchpad\xgb_trees_final.json", "w") as f:
    f.write(compact_json)

def predict_compact(trees, base_score, x_row):
    total = base_score
    for tree in trees:
        idx = 0
        while "leaf" not in tree[idx]:
            node = tree[idx]
            v = x_row[node["f"]]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                go_left = node["m"] == 1
            else:
                go_left = v < node["t"]
            idx = node["l"] if go_left else node["r"]
        total += tree[idx]["leaf"]
    return total

DATA_PATH = r"C:\Users\AMIN4PC\Desktop\Results_NewDataset_Full\Optimization - NewDataset (clean, all rows).xlsx"
dataset = load_data(file_path=DATA_PATH, extra_features=["NS size(nm)"])
X = dataset.X[feature_names]

# Worst-case validation: exact float32-quantized training values (adversarial for boundary precision)
Xf32 = X.values.astype(np.float32)
official_f32 = best_model.predict(Xf32)

diffs = []
for i in range(len(X)):
    manual = predict_compact(compact_trees, base_score, Xf32[i].tolist())
    diffs.append(abs(manual - official_f32[i]))
diffs = np.array(diffs)
print()
print("=== Validation vs float32-quantized inputs (worst case, exact training values) ===")
print("max diff:", diffs.max())
print("mean diff:", diffs.mean())

# Realistic case: raw float64 inputs (as typed by a user), compare to float32-based official
diffs64 = []
for i in range(len(X)):
    manual = predict_compact(compact_trees, base_score, X.iloc[i].tolist())
    diffs64.append(abs(manual - official_f32[i]))
diffs64 = np.array(diffs64)
print()
print("=== Validation vs raw float64 inputs (realistic) ===")
print("max diff:", diffs64.max())
print("mean diff:", diffs64.mean())

print()
print("R2 of manual vs official (float64 case):")
from sklearn.metrics import r2_score
manual_all = np.array([predict_compact(compact_trees, base_score, X.iloc[i].tolist()) for i in range(len(X))])
print(r2_score(official_f32, manual_all))
