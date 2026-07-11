import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import json
import joblib
import pandas as pd

from tensorflow import keras
from sklearn.metrics import classification_report, f1_score, confusion_matrix

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

# -----------------------------------------------------------------------
# 2. Run both models on the full test set
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
#    Shows the AE's standalone detection ability before XGBoost runs
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("STAGE 1 — AUTOENCODER ALONE (binary detection)")
print("="*65)

y_binary   = (y_test != BENIGN_LABEL).astype(int)
ae_binary  = flagged.astype(int)

print(classification_report(
    y_binary, ae_binary,
    target_names=["BENIGN", "Attack"],
    digits=4
))

# per-class flagging rate — shows which attack types AE catches
print(f"{'Class':<30} {'Count':>8} {'Flagged':>10}")
print("-" * 52)
for cid in np.unique(y_test):
    name = label_encoder.inverse_transform([cid])[0]
    mask = (y_test == cid)
    pct  = flagged[mask].mean() * 100
    print(f"{name:<30} {mask.sum():>8} {pct:>9.1f}%")

ae_f1 = f1_score(y_binary, ae_binary, average="macro")
print(f"\nAE Binary Macro F1: {ae_f1:.4f}")

# -----------------------------------------------------------------------
# 4. Stage 2 — XGBoost alone (15-class, full test set, no AE)
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("STAGE 2 — XGBOOST ALONE (15-class, full test set)")
print("="*65)

print(classification_report(
    y_test,
    xgb_preds,
    target_names=label_encoder.classes_,
    digits=4
))

xgb_f1 = f1_score(y_test, xgb_preds, average="macro")
print(f"XGBoost Macro F1: {xgb_f1:.4f}")

# -----------------------------------------------------------------------
# 5. Combined pipeline — confidence-gated AE + XGBoost (FINAL MODEL)
#    Decision: flag Unknown Anomaly only when BOTH conditions hold:
#      (a) AE reconstruction error > threshold
#      (b) XGBoost BENIGN confidence < CONFIDENCE_GATE (0.90)
#    Otherwise use XGBoost's predicted class directly
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("FINAL PIPELINE — CONFIDENCE-GATED AE + XGBOOST (gate=0.90)")
print("="*65)

unknown_mask = flagged & (xgb_preds == BENIGN_LABEL) & (benign_conf < CONFIDENCE_GATE)

print(f"\nFlows flagged as Unknown Anomaly: {unknown_mask.sum()}")

if unknown_mask.sum() > 0:
    true_of_unknowns = label_encoder.inverse_transform(y_test[unknown_mask])
    print("True labels of Unknown Anomaly flows:")
    print(pd.Series(true_of_unknowns).value_counts())

# exclude Unknown Anomaly from 15-class report
# (they were deliberately not assigned a class label)
non_unknown = ~unknown_mask

print(f"\n--- 15-Class Report (excluding {unknown_mask.sum()} Unknown Anomaly flows) ---")
print(classification_report(
    y_test[non_unknown],
    xgb_preds[non_unknown],
    target_names=label_encoder.classes_,
    digits=4
))

combined_f1 = f1_score(
    y_test[non_unknown],
    xgb_preds[non_unknown],
    average="macro"
)
print(f"Combined Pipeline Macro F1 (excl. Unknown Anomaly): {combined_f1:.4f}")
print(f"Flows evaluated: {non_unknown.sum():,} / {len(y_test):,}")

# -----------------------------------------------------------------------
# 6. Summary — all three versions side by side
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"{'Version':<45} {'Macro F1':>10}")
print("-" * 57)
print(f"{'AE alone (binary)':<45} {ae_f1:>10.4f}")
print(f"{'XGBoost alone (15-class)':<45} {xgb_f1:>10.4f}")
print(f"{'Confidence-gated AE + XGBoost (final)':<45} {combined_f1:>10.4f}")
print(f"\nNote: AE binary F1 and 15-class F1 are not directly comparable")
print(f"(binary has 2 classes, 15-class has 15). Combined pipeline F1")
print(f"reflects the complete system on {non_unknown.sum():,} flows where a")
print(f"definitive class label was assigned.")