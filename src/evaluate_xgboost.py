import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import json
import joblib
import pandas as pd

from tensorflow import keras
from sklearn.metrics import classification_report, f1_score

start_label = "CONFIDENCE-GATED PIPELINE (XGBoost + AE, gate=0.90)"

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

BENIGN_LABEL = label_encoder.transform(["BENIGN"])[0]

# -----------------------------------------------------------------------
# 2. Run both models
# -----------------------------------------------------------------------
xgb_preds   = xgb_model.predict(X_test)
xgb_probs   = xgb_model.predict_proba(X_test)
benign_conf = xgb_probs[:, BENIGN_LABEL]   # XGBoost's confidence specifically in BENIGN

recon    = autoencoder.predict(X_test, batch_size=512, verbose=0)
test_mse = np.mean(np.square(X_test - recon), axis=1)
flagged  = test_mse > threshold

# -----------------------------------------------------------------------
# 3. Confidence-gated decision logic
#    Only override to Unknown Anomaly if XGBoost itself is unsure
#    about BENIGN (confidence < 0.90). If XGBoost is very confident
#    it's BENIGN, trust XGBoost and ignore the AE's flag.
# -----------------------------------------------------------------------
CONFIDENCE_GATE = 0.90

unknown_mask = flagged & (xgb_preds == BENIGN_LABEL) & (benign_conf < CONFIDENCE_GATE)

print(f"Flows flagged as Unknown Anomaly: {unknown_mask.sum()}")

if unknown_mask.sum() > 0:
    true_labels_of_unknowns = label_encoder.inverse_transform(y_test[unknown_mask])
    print("\nTrue labels of flows flagged as Unknown Anomaly:")
    print(pd.Series(true_labels_of_unknowns).value_counts())

# -----------------------------------------------------------------------
# 4. Combined pipeline evaluation (excluding Unknown Anomaly flows)
#    Same format as the original combined pipeline report
# -----------------------------------------------------------------------
print(f"\n--- {start_label} ---")

final_preds   = xgb_preds.copy()
non_unknown   = ~unknown_mask

print(classification_report(
    y_test[non_unknown],
    final_preds[non_unknown],
    target_names=label_encoder.classes_,
    digits=4
))

combined_macro_f1 = f1_score(
    y_test[non_unknown],
    final_preds[non_unknown],
    average="macro"
)
print(f"Combined Macro F1 (excluding Unknown Anomaly): {combined_macro_f1:.4f}")
print(f"Flows evaluated: {non_unknown.sum()} / {len(y_test)}")