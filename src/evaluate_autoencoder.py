import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
import json
import joblib

from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

start = time.time()

# -----------------------------------------------------------------------
# 1. Load test data, model, threshold
# -----------------------------------------------------------------------
print("Loading...")
t0 = time.time()

X_test        = np.load("processed/X_test.npy")
y_test        = np.load("processed/y_test.npy")
autoencoder   = keras.models.load_model("models/autoencoder.keras")
label_encoder = joblib.load("models/label_encoder.joblib")

with open("models/threshold.json") as f:
    data = json.load(f)

threshold = data["threshold"]
print(f"Loaded threshold (P{data['percentile']}): {threshold:.6f}  ({time.time()-t0:.1f}s)")

BENIGN_LABEL   = label_encoder.transform(["BENIGN"])[0]
y_test_binary  = (y_test != BENIGN_LABEL).astype(int)  # 0=benign, 1=attack

# -----------------------------------------------------------------------
# 2. Compute test reconstruction errors
# -----------------------------------------------------------------------
print("\nComputing reconstruction errors...")
t0 = time.time()

recon    = autoencoder.predict(X_test, batch_size=512, verbose=0)
test_mse = np.mean(np.square(X_test - recon), axis=1)

print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 3. Compare percentile thresholds
#    Shows sensitivity of detection to threshold choice.
#    Best F1 determines which threshold to use for final evaluation.
# -----------------------------------------------------------------------
print("\n--- Threshold Comparison ---")

# these values come from train_autoencoder.py output (BENIGN val errors)
thresholds = {
    "P90": 0.000022,
    "P95": 0.000044,
    "P97": 0.000072,
    "P99": 0.000189
}

best_f1        = 0
best_threshold = None
best_name      = None

for name, t in thresholds.items():
    preds              = (test_mse > t).astype(int)
    tn, fp, fn, tp     = confusion_matrix(y_test_binary, preds).ravel()
    fpr                = fp / (fp + tn)
    recall             = tp / (tp + fn)
    precision          = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1                 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f"{name}: Threshold={t:.6f} | FPR={fpr:.4f} | Recall={recall:.4f} | F1={f1:.4f}")

    if f1 > best_f1:
        best_f1        = f1
        best_threshold = t
        best_name      = name

print(f"\nSelected: {best_name} (F1={best_f1:.4f})")

# -----------------------------------------------------------------------
# 4. Per-class flagging rates using the best threshold
#    BENIGN should be ~5% (by P95 definition)
#    High flagging on attack classes = AE catches them well
#    Low flagging = XGBoost (Stage 2) must carry the load for those
# -----------------------------------------------------------------------
flagged = (test_mse > best_threshold)

print(f"\n{'Class':<30} {'Count':>8} {'Flagged':>10}")
print("-" * 52)

for class_id in np.unique(y_test):
    name = label_encoder.inverse_transform([class_id])[0]
    mask = (y_test == class_id)
    pct  = flagged[mask].mean() * 100
    print(f"{name:<30} {mask.sum():>8} {pct:>9.1f}%")

# -----------------------------------------------------------------------
# 5. Binary classification report
# -----------------------------------------------------------------------
print("\n--- Binary Detection Report ---")
print(classification_report(
    y_test_binary,
    flagged.astype(int),
    target_names=["BENIGN", "Attack"],
    digits=4
))

print(f"Total time: {time.time()-start:.1f}s")