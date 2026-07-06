import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import json
import joblib

from tensorflow import keras
from sklearn.metrics import classification_report, f1_score

# -----------------------------------------------------------------------
# 1. Load everything
# -----------------------------------------------------------------------
X_test        = np.load("processed/X_test.npy")
y_test        = np.load("processed/y_test.npy")
xgb_model     = joblib.load("models/xgboost.joblib")
autoencoder   = keras.models.load_model("models/autoencoder.keras")
label_encoder = joblib.load("models/label_encoder.joblib")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

BENIGN_LABEL  = label_encoder.transform(["BENIGN"])[0]
WRONG_LABEL   = (BENIGN_LABEL + 1) % len(label_encoder.classes_)  # any non-benign label, used to mark misses

xgb_preds   = xgb_model.predict(X_test)
xgb_probs   = xgb_model.predict_proba(X_test)
benign_conf = xgb_probs[:, BENIGN_LABEL]

recon    = autoencoder.predict(X_test, batch_size=512, verbose=0)
test_mse = np.mean(np.square(X_test - recon), axis=1)
flagged  = test_mse > threshold

# -----------------------------------------------------------------------
# 2. Build a TRUE combined predictions array on the FULL test set
#    No flows are excluded from the report — Unknown Anomaly flows
#    are scored against the true 15-class label:
#      - if true label is BENIGN  -> counted as WRONG (false alarm)
#      - if true label is attack  -> counted as CORRECT (AE caught it)
#    This avoids shrinking the denominator and gives the AE neither
#    unearned credit nor unearned exclusion.
# -----------------------------------------------------------------------
def build_combined_preds(unknown_mask):
    preds = xgb_preds.copy()

    false_alarm_mask = unknown_mask & (y_test == BENIGN_LABEL)
    preds[false_alarm_mask] = WRONG_LABEL          # penalize: AE wrongly flagged real BENIGN

    correct_catch_mask = unknown_mask & (y_test != BENIGN_LABEL)
    preds[correct_catch_mask] = y_test[correct_catch_mask]   # credit: AE caught a real attack

    return preds

original_unknown_mask = flagged & (xgb_preds == BENIGN_LABEL)
gated_unknown_mask     = flagged & (xgb_preds == BENIGN_LABEL) & (benign_conf < 0.90)

# -----------------------------------------------------------------------
# 3. Score all three versions on the SAME full test set, same format
# -----------------------------------------------------------------------
print("--- XGBOOST ALONE (full test set, no AE) ---")
print(classification_report(y_test, xgb_preds, target_names=label_encoder.classes_, digits=4))
xgb_f1 = f1_score(y_test, xgb_preds, average="macro")
print(f"Macro F1: {xgb_f1:.4f}\n")

print("--- COMBINED PIPELINE (original AE logic, P95, no gate) — full test set ---")
preds_orig = build_combined_preds(original_unknown_mask)
print(classification_report(y_test, preds_orig, target_names=label_encoder.classes_, digits=4))
orig_f1 = f1_score(y_test, preds_orig, average="macro")
print(f"Macro F1: {orig_f1:.4f}\n")

print("--- COMBINED PIPELINE (confidence-gated, gate=0.90) — full test set ---")
preds_gated = build_combined_preds(gated_unknown_mask)
print(classification_report(y_test, preds_gated, target_names=label_encoder.classes_, digits=4))
gated_f1 = f1_score(y_test, preds_gated, average="macro")
print(f"Macro F1: {gated_f1:.4f}\n")

# -----------------------------------------------------------------------
# 4. Side-by-side summary
# -----------------------------------------------------------------------
print("--- SUMMARY (all scored on same 419,376 flows, no exclusions) ---")
print(f"{'Version':<45}{'Macro F1':>10}")
print("-" * 55)
print(f"{'XGBoost alone':<45}{xgb_f1:>10.4f}")
print(f"{'Combined (original AE logic)':<45}{orig_f1:>10.4f}")
print(f"{'Combined (confidence-gated)':<45}{gated_f1:>10.4f}")