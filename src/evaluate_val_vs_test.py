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

X_val         = np.load("processed/X_val.npy")
y_val         = np.load("processed/y_val.npy")
X_test        = np.load("processed/X_test.npy")
y_test        = np.load("processed/y_test.npy")
xgb_model     = joblib.load("models/xgboost.joblib")
autoencoder   = keras.models.load_model("models/autoencoder.keras")
label_encoder = joblib.load("models/label_encoder.joblib")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

BENIGN_LABEL    = label_encoder.transform(["BENIGN"])[0]
CONFIDENCE_GATE = 0.90
WRONG_LABEL = -1
VALID_LABELS = list(range(len(label_encoder.classes_)))


# -----------------------------------------------------------------------
# 2. Run both models on val and test
# -----------------------------------------------------------------------
print("Running models on validation set...")
xgb_preds_val   = xgb_model.predict(X_val)
xgb_probs_val   = xgb_model.predict_proba(X_val)
benign_conf_val = xgb_probs_val[:, BENIGN_LABEL]
recon_val       = autoencoder.predict(X_val, batch_size=512, verbose=0)
mse_val         = np.mean(np.square(X_val - recon_val), axis=1)
flagged_val     = mse_val > threshold

print("Running models on test set...")
xgb_preds_test   = xgb_model.predict(X_test)
xgb_probs_test   = xgb_model.predict_proba(X_test)
benign_conf_test = xgb_probs_test[:, BENIGN_LABEL]
recon_test       = autoencoder.predict(X_test, batch_size=512, verbose=0)
mse_test         = np.mean(np.square(X_test - recon_test), axis=1)
flagged_test     = mse_test > threshold

# -----------------------------------------------------------------------
# Helper: build honest combined predictions
# Unknown Anomaly: correct if true label is attack, wrong if BENIGN
# -----------------------------------------------------------------------
def build_combined(xgb_preds, flagged, benign_conf, y_true, gate):
    unknown_mask = (
        flagged &
        (xgb_preds == BENIGN_LABEL) &
        (benign_conf < gate)
    )

    preds = xgb_preds.copy()

    correct_catch = unknown_mask & (y_true != BENIGN_LABEL)
    preds[correct_catch] = y_true[correct_catch]

    false_alarm = unknown_mask & (y_true == BENIGN_LABEL)
    preds[false_alarm] = WRONG_LABEL

    return preds, unknown_mask

# -----------------------------------------------------------------------
# 3. Compute macro F1 for all four versions on val and test
# -----------------------------------------------------------------------
def get_f1(y_true, preds):
    return f1_score(
        y_true,
        preds,
        labels=VALID_LABELS,
        average="macro"
    )


# AE binary F1 (val and test)
ae_f1_val  = f1_score((y_val  != BENIGN_LABEL).astype(int), flagged_val.astype(int),  average="macro")
ae_f1_test = f1_score((y_test != BENIGN_LABEL).astype(int), flagged_test.astype(int), average="macro")

# XGBoost standalone
xgb_f1_val  = get_f1(y_val,  xgb_preds_val)
xgb_f1_test = get_f1(y_test, xgb_preds_test)

# Naive combined (no gate)
naive_val,  mask_naive_val  = build_combined(xgb_preds_val,  flagged_val,  benign_conf_val,  y_val,  1.0)
naive_test, mask_naive_test = build_combined(xgb_preds_test, flagged_test, benign_conf_test, y_test, 1.0)
naive_f1_val  = get_f1(y_val,  naive_val)
naive_f1_test = get_f1(y_test, naive_test)

# Confidence-gated combined (final)
gated_val,  mask_gated_val  = build_combined(xgb_preds_val,  flagged_val,  benign_conf_val,  y_val,  CONFIDENCE_GATE)
gated_test, mask_gated_test = build_combined(xgb_preds_test, flagged_test, benign_conf_test, y_test, CONFIDENCE_GATE)
gated_f1_val  = get_f1(y_val,  gated_val)
gated_f1_test = get_f1(y_test, gated_test)

# -----------------------------------------------------------------------
# 4. Summary table — val vs test for all four versions
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("SUMMARY: VALIDATION vs TEST — ALL FOUR PIPELINE VERSIONS")
print("(All honest: same scoring rule, no flow exclusions)")
print("="*70)
print(f"{'Version':<40} {'Val F1':>8} {'Test F1':>8} {'Gap':>8}")
print("-"*66)
for label, vf1, tf1 in [
    ("AE alone (binary)",           ae_f1_val,    ae_f1_test),
    ("XGBoost alone",               xgb_f1_val,   xgb_f1_test),
    ("Naive AE+XGB (no gate)",      naive_f1_val,  naive_f1_test),
    ("Confidence-gated (FINAL)",    gated_f1_val,  gated_f1_test),
]:
    gap = tf1 - vf1
    flag = " ⚠" if abs(gap) > 0.02 else ""
    print(f"{label:<40} {vf1:>8.4f} {tf1:>8.4f} {gap:>+8.4f}{flag}")
print()
print("Gap = Test F1 - Val F1. Negative = model did worse on test.")
print("Gap > 0.02 flagged with ⚠ as potential overfitting concern.")

# -----------------------------------------------------------------------
# 5. Per-class F1 comparison: XGBoost val vs test
#    This is the most informative overfitting check
# -----------------------------------------------------------------------

print("\n" + "="*70)
print("PER-CLASS F1: XGBOOST STANDALONE — VAL vs TEST")
print("="*70)

val_f1s = f1_score(
    y_val,
    xgb_preds_val,
    labels=VALID_LABELS,
    average=None
)

test_f1s = f1_score(
    y_test,
    xgb_preds_test,
    labels=VALID_LABELS,
    average=None
)

print(f"{'Class':<35} {'Val F1':>8} {'Test F1':>8} {'Gap':>8}")
print("-"*63)

for i, cls in enumerate(label_encoder.classes_):
    gap = test_f1s[i] - val_f1s[i]
    flag = " ⚠" if abs(gap) > 0.05 else ""
    print(f"{cls:<35} {val_f1s[i]:>8.4f} {test_f1s[i]:>8.4f} {gap:>+8.4f}{flag}")

print(f"\n{'Macro Average':<35} {xgb_f1_val:>8.4f} {xgb_f1_test:>8.4f} {xgb_f1_test-xgb_f1_val:>+8.4f}")

# -----------------------------------------------------------------------
# 6. Per-class F1 comparison: Confidence-gated pipeline val vs test
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("PER-CLASS F1: CONFIDENCE-GATED PIPELINE — VAL vs TEST")
print("="*70)

val_f1s_g = f1_score(
    y_val,
    gated_val,
    labels=VALID_LABELS,
    average=None
)

test_f1s_g = f1_score(
    y_test,
    gated_test,
    labels=VALID_LABELS,
    average=None
)

print(f"{'Class':<35} {'Val F1':>8} {'Test F1':>8} {'Gap':>8}")
print("-"*63)
for i, cls in enumerate(label_encoder.classes_):
    gap = test_f1s_g[i] - val_f1s_g[i]
    flag = " ⚠" if abs(gap) > 0.05 else ""
    print(f"{cls:<35} {val_f1s_g[i]:>8.4f} {test_f1s_g[i]:>8.4f} {gap:>+8.4f}{flag}")

print(f"\n{'Macro Average':<35} {gated_f1_val:>8.4f} {gated_f1_test:>8.4f} {gated_f1_test-gated_f1_val:>+8.4f}")
# -----------------------------------------------------------------------
# 7. Full classification report on validation — confidence-gated pipeline
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("FULL REPORT: CONFIDENCE-GATED PIPELINE ON VALIDATION SET")
print("="*70)
print(f"Unknown Anomaly on val: {mask_gated_val.sum()}")

print(classification_report(
    y_val,
    gated_val,
    labels=VALID_LABELS,
    target_names=label_encoder.classes_,
    digits=4,
    zero_division=0
))

print("\n" + "="*70)
print("FULL REPORT: CONFIDENCE-GATED PIPELINE ON TEST SET")
print("="*70)
print(f"Unknown Anomaly on test: {mask_gated_test.sum()}")

print(classification_report(
    y_test,
    gated_test,
    labels=VALID_LABELS,
    target_names=label_encoder.classes_,
    digits=4,
    zero_division=0
))
