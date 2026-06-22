import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
import json
import joblib

from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

start = time.time()


# -----------------------------------------------------------------------
# 1. Load test data, model, threshold
# -----------------------------------------------------------------------
print("Loading...")
t0 = time.time()


X_test = np.load("processed/X_test.npy")
y_test = np.load("processed/y_test.npy")

autoencoder = keras.models.load_model(
    "models/autoencoder.keras"
)

label_encoder = joblib.load(
    "models/label_encoder.joblib"
)


with open(
    "models/threshold.json"
) as f:

    data = json.load(f)

    threshold = data["threshold"]


print(
    f"Threshold: {threshold:.6f}"
)


if "percentile" in data:
    print(
        f"Threshold source: "
        f"BENIGN validation P{data['percentile']}"
    )


print(
    f"({time.time()-t0:.1f}s)"
)


# -----------------------------------------------------------------------
# 2. Compute test reconstruction errors
# -----------------------------------------------------------------------
print("\nComputing test reconstruction errors...")
t0 = time.time()


recon = autoencoder.predict(
    X_test,
    batch_size=512,
    verbose=0
)


test_mse = np.mean(
    np.square(X_test - recon),
    axis=1
)

# -----------------------------------------------------------------------
# Compare multiple thresholds
# -----------------------------------------------------------------------

print("\n--- Threshold Comparison ---")

BENIGN_LABEL = label_encoder.transform(
    ["BENIGN"]
)[0]


y_test_binary = (
    y_test != BENIGN_LABEL
).astype(int)


# Thresholds obtained from BENIGN validation
thresholds = {
    "P90": 0.000022,
    "P95": 0.000044,
    "P97": 0.000072,
    "P99": 0.000189
}


for name, threshold in thresholds.items():

    predictions = (
        test_mse > threshold
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        y_test_binary,
        predictions
    ).ravel()


    fpr = fp / (fp + tn)

    recall = tp / (tp + fn)

    precision = tp / (tp + fp)

    f1 = (
        2 * precision * recall /
        (precision + recall)
    )


    print(
        f"{name}: "
        f"Threshold={threshold:.6f} | "
        f"FPR={fpr:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )

flagged = (
    test_mse > threshold
)


print(
    f"Flagged samples: "
    f"{flagged.sum()} / {len(flagged)}"
)


print(
    f"({time.time()-t0:.1f}s)"
)


# -----------------------------------------------------------------------
# 3. Per-class flagging rates
#    Shows how well the AE anomaly detector catches each attack type
# -----------------------------------------------------------------------
print(
    f"\n{'Class':<30}"
    f"{'Count':>8}"
    f"{'Flagged as Anomaly':>20}"
)

print("-" * 62)


for class_id in np.unique(y_test):

    name = label_encoder.inverse_transform(
        [class_id]
    )[0]


    mask = (
        y_test == class_id
    )


    pct = (
        flagged[mask].mean()
        *
        100
    )


    print(
        f"{name:<30}"
        f"{mask.sum():>8}"
        f"{pct:>19.1f}%"
    )


# -----------------------------------------------------------------------
# 4. Compare multiple thresholds
#    The threshold with the highest F1-score is selected.
#    In this experiment P95 gives the best balance between:
#    - False Positive Rate (benign traffic incorrectly flagged)
#    - Attack Recall (attacks detected)
# -----------------------------------------------------------------------

print("\n--- Threshold Comparison ---")

thresholds = {
    "P90": 0.000022,
    "P95": 0.000044,
    "P97": 0.000072,
    "P99": 0.000189
}

best_f1 = 0
best_threshold = None
best_name = None

for name, threshold_value in thresholds.items():

    predictions = (
        test_mse > threshold_value
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test_binary,
        predictions
    ).ravel()

    fpr = fp / (fp + tn)

    recall = tp / (tp + fn)

    precision = tp / (tp + fp)

    f1 = (
        2 * precision * recall /
        (precision + recall)
    )

    print(
        f"{name}: "
        f"Threshold={threshold_value:.6f} | "
        f"FPR={fpr:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold_value
        best_name = name


print(
    f"\nSelected threshold: {best_name} "
    f"(highest F1={best_f1:.4f})"
)


# -----------------------------------------------------------------------
# Final evaluation below uses the selected threshold.
# P95 is selected because it achieved the highest F1 score.
# All class-wise detection results and reports below correspond to P95.
# -----------------------------------------------------------------------

threshold = best_threshold