import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF info/warning logs

import numpy as np
import time
import json
import joblib

from tensorflow import keras
from tensorflow.keras import layers, callbacks

start = time.time()


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
# Threshold is calculated from BENIGN validation reconstruction errors.
#
# Lower percentile:
#   - catches more attacks
#   - more false alarms
#
# Higher percentile:
#   - fewer false alarms
#   - may miss attacks
# -----------------------------------------------------------------------
THRESHOLD_PERCENTILE = 95



# -----------------------------------------------------------------------
# 1. Load preprocessed arrays
# -----------------------------------------------------------------------
print("Loading data...")
t0 = time.time()

X_train = np.load("processed/X_train.npy")
y_train = np.load("processed/y_train.npy")
X_val   = np.load("processed/X_val.npy")
y_val   = np.load("processed/y_val.npy")

label_encoder = joblib.load("models/label_encoder.joblib")

print(f"Train: {X_train.shape}, Val: {X_val.shape}  ({time.time()-t0:.1f}s)")


# -----------------------------------------------------------------------
# 2. Filter train to BENIGN only
#    Autoencoder learns what "normal" looks like — never sees attacks
#    during training. Anomaly = high reconstruction error at inference.
# -----------------------------------------------------------------------
print("\nFiltering train to BENIGN only...")
t0 = time.time()

BENIGN_LABEL = label_encoder.transform(["BENIGN"])[0]

X_train_benign = X_train[y_train == BENIGN_LABEL]

print(f"Benign train flows: {X_train_benign.shape}  ({time.time()-t0:.1f}s)")


# -----------------------------------------------------------------------
# 3. Build autoencoder
#    Dense architecture: input -> 64 -> 32 (bottleneck) -> 64 -> input
#    Sigmoid on output since MinMaxScaler ensures all features in [0,1]
# -----------------------------------------------------------------------
print("\nBuilding autoencoder...")

input_dim = X_train_benign.shape[1]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),              # bottleneck
    layers.Dense(64, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")     # reconstruct input
])

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()


# -----------------------------------------------------------------------
# 4. Train with early stopping on validation loss
#    We use X_val (all classes) for validation loss monitoring.
#
#    The AE sees attack flows only during validation calculation.
#    It is not trained on attacks.
#
#    patience=5:
#       stop if val_loss does not improve for 5 epochs.
#
#    restore_best_weights:
#       return to the best epoch automatically.
# -----------------------------------------------------------------------
print("\nTraining...")
t0 = time.time()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


history = autoencoder.fit(
    X_train_benign,
    X_train_benign,                 # reconstruction task: input = target
    epochs=100,
    batch_size=256,
    validation_data=(X_val, X_val),
    callbacks=[early_stop],
    verbose=1
)


epochs_ran = len(history.history["loss"])

print(f"Stopped at epoch {epochs_ran}  ({time.time()-t0:.1f}s)")


# -----------------------------------------------------------------------
# 5. Compute reconstruction errors on validation set
#    Used to calculate the anomaly threshold.
#
#    Threshold is learned only from BENIGN validation errors.
# -----------------------------------------------------------------------
print("\nComputing val reconstruction errors...")
t0 = time.time()


val_recon = autoencoder.predict(
    X_val,
    batch_size=512,
    verbose=0
)


val_mse = np.mean(
    np.square(X_val - val_recon),
    axis=1
)


print(
    f"Val MSE — mean: {val_mse.mean():.6f}, "
    f"max: {val_mse.max():.6f}"
)

print(f"({time.time()-t0:.1f}s)")


# -----------------------------------------------------------------------
# 6. Calculate anomaly threshold
#
#    Autoencoder is trained only on BENIGN traffic.
#    Therefore threshold should represent the boundary of normal traffic.
#
#    Using benign-only reconstruction errors prevents the threshold from
#    being influenced by attack samples.
#
#    Example:
#       95th percentile means:
#       - 95% of benign validation flows are accepted
#       - highest 5% benign errors are flagged as suspicious
# -----------------------------------------------------------------------
print("\nCalculating anomaly threshold...")
t0 = time.time()


benign_mse = val_mse[
    y_val == BENIGN_LABEL
]


print("Benign reconstruction error percentiles:")

for p in [90, 95, 97, 99]:
    print(
        f"P{p}: {np.percentile(benign_mse, p):.6f}"
    )


threshold = np.percentile(
    benign_mse,
    THRESHOLD_PERCENTILE
)


if threshold <= 0:
    raise ValueError(
        "Invalid threshold calculated."
    )


print(
    f"\nSelected threshold (P{THRESHOLD_PERCENTILE}): "
    f"{threshold:.6f}"
)

print(f"({time.time()-t0:.1f}s)")


# -----------------------------------------------------------------------
# 7. Save model and threshold
# -----------------------------------------------------------------------
print("\nSaving...")
t0 = time.time()


autoencoder.save(
    "models/autoencoder.keras"
)


with open(
    "models/threshold.json",
    "w"
) as f:

    json.dump(
        {
            "threshold": float(threshold),
            "percentile": THRESHOLD_PERCENTILE,
            "epochs_trained": epochs_ran
        },
        f,
        indent=2
    )


print(f"({time.time()-t0:.1f}s)")
print(f"\nTotal time: {time.time()-start:.1f}s")