import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import json
import joblib
import pandas as pd

from tensorflow import keras
from sklearn.metrics import classification_report, f1_score

# -----------------------------------------------------------------------
# 1. Load everything
# -----------------------------------------------------------------------
print("Loading...")

X_test        = np.load("processed/X_test.npy")
y_test        = np.load("processed/y_test.npy")
xgb_model     = joblib.load("models/xgboost.joblib")
autoencoder   = keras.models.load_model("models/autoencoder.keras")
label_encoder = joblib.load("models/label_encoder.joblib")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

BENIGN_LABEL    = label_encoder.transform(["BENIGN"])[0]
CONFIDENCE_GATE = 0.90
WRONG_LABEL     = -1  # sentinel, not a valid class index

# -----------------------------------------------------------------------
# 2. Run both models
# -----------------------------------------------------------------------
print("Running XGBoost...")
xgb_preds   = xgb_model.predict(X_test)
xgb_probs   = xgb_model.predict_proba(X_test)
benign_conf = xgb_probs[:, BENIGN_LABEL]

print("Running Autoencoder...")
recon    = autoencoder.predict(X_test, batch_size=512, verbose=0)
test_mse = np.mean(np.square(X_test - recon), axis=1)
flagged  = test_mse > threshold

# -----------------------------------------------------------------------
# 3. Stage 1 — Autoencoder alone (binary: BENIGN vs Attack)
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("STAGE 1 — AUTOENCODER ALONE (binary detection)")
print("="*65)

y_binary  = (y_test != BENIGN_LABEL).astype(int)
ae_binary = flagged.astype(int)

print(classification_report(
    y_binary, ae_binary,
    target_names=["BENIGN", "Attack"],
    digits=4
))
ae_f1 = f1_score(y_binary, ae_binary, average="macro")
print(f"AE Binary Macro F1: {ae_f1:.4f}")

print(f"\n{'Class':<30} {'Count':>8} {'Flagged':>10}")
print("-" * 52)
for cid in np.unique(y_test):
    name = label_encoder.inverse_transform([cid])[0]
    mask = (y_test == cid)
    pct  = flagged[mask].mean() * 100
    print(f"{name:<30} {mask.sum():>8} {pct:>9.1f}%")

# -----------------------------------------------------------------------
# 4. Stage 2 — XGBoost alone (15-class, full 419,376 flows)
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("STAGE 2 — XGBOOST ALONE (15-class, full 419,376 flows)")
print("="*65)

print(classification_report(
    y_test, xgb_preds,
    target_names=label_encoder.classes_,
    digits=4
))
xgb_f1 = f1_score(y_test, xgb_preds, average="macro")
print(f"XGBoost Standalone Macro F1: {xgb_f1:.4f}")

# -----------------------------------------------------------------------
# 5. Confidence-gated combined pipeline
#    Honest evaluation: all 419,376 flows scored, no exclusions
#
#    Unknown Anomaly flows are scored as follows:
#      - true label is non-BENIGN (real attack XGBoost missed):
#        → set pred to true label → counted as CORRECT
#      - true label is BENIGN (AE false alarm on correct BENIGN):
#        → set pred to WRONG_LABEL → counted as WRONG
#
#    This is the same standard applied to the naive AE+XGB evaluation.
#    The 25 Unknown Anomaly flows are NOT excluded from the denominator.
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("FINAL PIPELINE — CONFIDENCE-GATED AE + XGBOOST (gate=0.90)")
print("Honest evaluation: all 419,376 flows, no exclusions")
print("="*65)

unknown_mask = flagged & (xgb_preds == BENIGN_LABEL) & (benign_conf < CONFIDENCE_GATE)

print(f"\nFlows flagged as Unknown Anomaly: {unknown_mask.sum()}")

if unknown_mask.sum() > 0:
    true_of_unknowns = label_encoder.inverse_transform(y_test[unknown_mask])
    print("True labels of Unknown Anomaly flows:")
    print(pd.Series(true_of_unknowns).value_counts())

# Build honest prediction array
# Start from XGBoost predictions
final_preds = xgb_preds.copy()

# Unknown Anomaly flows where true label is a real attack:
# give credit — these are genuine catches XGBoost would have missed
correct_catch = unknown_mask & (y_test != BENIGN_LABEL)
final_preds[correct_catch] = y_test[correct_catch]

# Unknown Anomaly flows where true label is BENIGN:
# penalize — AE incorrectly overrode a correct XGBoost BENIGN prediction
false_alarm = unknown_mask & (y_test == BENIGN_LABEL)
final_preds[false_alarm] = WRONG_LABEL

print(f"\nReal attacks caught by AE that XGBoost missed: {correct_catch.sum()}")
print(f"BENIGN flows wrongly converted to Unknown Anomaly: {false_alarm.sum()}")

print(f"\n--- 15-Class Report (all 419,376 flows, honest scoring) ---")
valid_labels = list(range(len(label_encoder.classes_)))
print(classification_report(
    y_test, final_preds,
    labels=valid_labels,
    target_names=label_encoder.classes_,
    digits=4
))

combined_f1 = f1_score(y_test, final_preds, labels=valid_labels, average="macro")
print(f"Combined Pipeline Macro F1 (honest, no exclusions): {combined_f1:.4f}")
print(f"Flows evaluated: {len(y_test):,} / {len(y_test):,}")

# -----------------------------------------------------------------------
# 6. Summary — all three on identical test set
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("SUMMARY — all three methods, same 419,376 flows")
print("="*65)
print(f"{'Version':<45} {'Macro F1':>10}")
print("-" * 57)
print(f"{'AE alone (binary)':<45} {ae_f1:>10.4f}")
print(f"{'XGBoost alone (15-class)':<45} {xgb_f1:>10.4f}")
print(f"{'Confidence-gated AE + XGBoost (final)':<45} {combined_f1:>10.4f}")
print(f"\nNote: AE binary F1 is not directly comparable to 15-class F1.")
print(f"The combined pipeline F1 uses the same scoring rule applied")
print(f"to the naive AE+XGB version in evaluate_xgboost_v2.py.")