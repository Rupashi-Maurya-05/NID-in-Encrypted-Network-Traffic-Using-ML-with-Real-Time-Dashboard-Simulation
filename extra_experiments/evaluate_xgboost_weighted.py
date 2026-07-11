import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
import json
import joblib

from tensorflow import keras
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pandas as pd

start = time.time()

# -----------------------------------------------------------------------
# 1. Load everything
# -----------------------------------------------------------------------
print("Loading...")
t0 = time.time()

X_test        = np.load("processed/X_test.npy")
y_test        = np.load("processed/y_test.npy")
xgb_model     = joblib.load("models/xgboost_weighted.joblib")
autoencoder   = keras.models.load_model("models/autoencoder.keras")
label_encoder = joblib.load("models/label_encoder.joblib")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

BENIGN_LABEL = label_encoder.transform(["BENIGN"])[0]
print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 2. XGBoost predictions
# -----------------------------------------------------------------------
print("\nRunning XGBoost predictions...")
t0 = time.time()

xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)
print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 3. XGBoost weighted standalone evaluation
# -----------------------------------------------------------------------
print("\n--- XGBoost Weighted Standalone (15-class) ---")
print(classification_report(
    y_test,
    xgb_preds,
    target_names=label_encoder.classes_,
    digits=4
))

macro_f1 = f1_score(y_test, xgb_preds, average="macro")
print(f"Macro F1: {macro_f1:.4f}")

# -----------------------------------------------------------------------
# 4. Quick comparison vs original unweighted results
#    Hardcoded from previous run for easy side-by-side reading
# -----------------------------------------------------------------------
print("\n--- Key Class Comparison: Weighted vs Unweighted ---")
print(f"{'Class':<30} {'Old F1':>8} {'New F1':>8} {'Change':>8}")
print("-" * 58)

old_f1s = {
    "BENIGN":                    0.9958,
    "Bot":                       0.6616,
    "DDoS":                      1.0000,
    "DoS GoldenEye":             0.9971,
    "DoS Hulk":                  0.9998,
    "DoS Slowhttptest":          0.9927,
    "DoS slowloris":             0.9991,
    "FTP-Patator":               0.9994,
    "Heartbleed":                0.6667,
    "Infiltration":              0.8750,
    "PortScan":                  0.9654,
    "SSH-Patator":               1.0000,
    "Web Attack - Brute Force":  0.7279,
    "Web Attack - Sql Injection":1.0000,
    "Web Attack - XSS":          0.4526,
}

# get new per-class f1 scores
from sklearn.metrics import f1_score as f1_per_class
new_f1s_array = f1_score(y_test, xgb_preds, average=None)

for i, class_name in enumerate(label_encoder.classes_):
    old = old_f1s.get(class_name, 0)
    new = new_f1s_array[i]
    change = new - old
    arrow = "↑" if change > 0.005 else ("↓" if change < -0.005 else "≈")
    print(f"{class_name:<30} {old:>8.4f} {new:>8.4f} {arrow} {change:>+.4f}")

# -----------------------------------------------------------------------
# 5. AE anomaly scores + two-stage decision logic
# -----------------------------------------------------------------------
print("\nRunning Autoencoder...")
t0 = time.time()

recon    = autoencoder.predict(X_test, batch_size=512, verbose=0)
test_mse = np.mean(np.square(X_test - recon), axis=1)
flagged  = test_mse > threshold

print(f"({time.time()-t0:.1f}s)")

unknown_mask = flagged & (xgb_preds == BENIGN_LABEL)
print(f"\nFlows flagged as Unknown Anomaly: {unknown_mask.sum()}")

if unknown_mask.sum() > 0:
    true_labels = label_encoder.inverse_transform(y_test[unknown_mask])
    print("True labels of Unknown Anomaly flows:")
    print(pd.Series(true_labels).value_counts())

# -----------------------------------------------------------------------
# 6. Combined pipeline evaluation
# -----------------------------------------------------------------------
print("\n--- Combined Pipeline (Weighted XGBoost + AE) ---")
non_unknown = ~unknown_mask

print(classification_report(
    y_test[non_unknown],
    xgb_preds[non_unknown],
    target_names=label_encoder.classes_,
    digits=4
))

combined_f1 = f1_score(y_test[non_unknown], xgb_preds[non_unknown], average="macro")
print(f"Combined Macro F1: {combined_f1:.4f}")
print(f"Previous combined Macro F1: 0.9197")
print(f"Change: {combined_f1 - 0.9197:+.4f}")

print(f"\nTotal time: {time.time()-start:.1f}s")