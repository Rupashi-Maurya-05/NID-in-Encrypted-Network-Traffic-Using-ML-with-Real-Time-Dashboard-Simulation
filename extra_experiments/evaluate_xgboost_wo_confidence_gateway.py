import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
import json
import joblib

from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd

start = time.time()

# -----------------------------------------------------------------------
# 1. Load everything
# -----------------------------------------------------------------------
print("Loading...")
t0 = time.time()

X_test        = np.load("processed/X_test.npy")
y_test        = np.load("processed/y_test.npy")
xgb_model     = joblib.load("models/xgboost.joblib")
autoencoder   = keras.models.load_model("models/autoencoder.keras")
label_encoder = joblib.load("models/label_encoder.joblib")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

print(f"({time.time()-t0:.1f}s)")

BENIGN_LABEL = label_encoder.transform(["BENIGN"])[0]

# -----------------------------------------------------------------------
# 2. XGBoost predictions on test set
# -----------------------------------------------------------------------
print("\nRunning XGBoost predictions...")
t0 = time.time()

xgb_preds  = xgb_model.predict(X_test)
xgb_probs  = xgb_model.predict_proba(X_test)
confidence = xgb_probs.max(axis=1)   # highest class probability per flow

print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 3. XGBoost standalone evaluation
# -----------------------------------------------------------------------
print("\n--- XGBoost Standalone (15-class) ---")
print(classification_report(
    y_test,
    xgb_preds,
    target_names=label_encoder.classes_,
    digits=4
))

macro_f1 = f1_score(y_test, xgb_preds, average="macro")
print(f"Macro F1: {macro_f1:.4f}")

# -----------------------------------------------------------------------
# 4. Autoencoder anomaly scores on test set
# -----------------------------------------------------------------------
print("\nRunning Autoencoder...")
t0 = time.time()

recon    = autoencoder.predict(X_test, batch_size=512, verbose=0)
test_mse = np.mean(np.square(X_test - recon), axis=1)
flagged  = test_mse > threshold

print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 5. Two-stage decision logic
#    if AE flags as anomaly AND XGBoost predicts BENIGN → Unknown Anomaly
#    otherwise → use XGBoost prediction
#
#    This catches flows the AE finds suspicious but XGBoost misclassifies
#    as benign — a safety net for novel/unknown attack patterns
# -----------------------------------------------------------------------
print("\nApplying two-stage decision logic...")

final_preds = xgb_preds.copy()

unknown_mask = flagged & (xgb_preds == BENIGN_LABEL)
print(f"Flows flagged as Unknown Anomaly: {unknown_mask.sum()}")

# for evaluation we map Unknown Anomaly back to a label
# we treat it as "not BENIGN" — correct if they're real attacks
# we'll report it separately rather than forcing it into 15 classes
unknown_indices = np.where(unknown_mask)[0]

# check what these "Unknown Anomaly" flows actually are
if unknown_mask.sum() > 0:
    true_labels_of_unknowns = label_encoder.inverse_transform(
        y_test[unknown_mask]
    )
    unknown_df = pd.Series(true_labels_of_unknowns).value_counts()
    print("\nTrue labels of flows flagged as Unknown Anomaly:")
    print(unknown_df)

# -----------------------------------------------------------------------
# 6. Combined pipeline evaluation
#    For this report, Unknown Anomaly flows are excluded from the
#    15-class report and counted separately — they were correctly
#    identified as suspicious even if we can't name the exact class
# -----------------------------------------------------------------------
print("\n--- Combined Pipeline (XGBoost + AE safety net) ---")

# only evaluate on flows where XGBoost gave a definitive class prediction
# (i.e. not overridden by Unknown Anomaly logic)
non_unknown_mask = ~unknown_mask

print(classification_report(
    y_test[non_unknown_mask],
    final_preds[non_unknown_mask],
    target_names=label_encoder.classes_,
    digits=4
))

combined_macro_f1 = f1_score(
    y_test[non_unknown_mask],
    final_preds[non_unknown_mask],
    average="macro"
)
print(f"Combined Macro F1 (excluding Unknown Anomaly): {combined_macro_f1:.4f}")
print(f"\nTotal time: {time.time()-start:.1f}s")