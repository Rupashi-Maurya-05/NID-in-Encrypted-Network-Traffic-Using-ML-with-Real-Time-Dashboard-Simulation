import numpy as np
import time
import json

from tensorflow import keras
from tensorflow.keras import layers

start = time.time()

# ----------------------------------------------------------------------
# 1. Load preprocessed data
# ----------------------------------------------------------------------
print("Loading data...")
t0 = time.time()

X_train = np.load("processed/X_train.npy")
y_train = np.load("processed/y_train.npy")

print(f"X_train shape: {X_train.shape}  (took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 2. Filter to BENIGN only
# (we need to know which encoded number BENIGN is — label_encoder
#  was saved alphabetically, BENIGN was first class -> encoded as 0)
# ----------------------------------------------------------------------
print("\nFiltering to BENIGN flows only...")
t0 = time.time()

BENIGN_LABEL = 0  # confirmed from label_encoder.classes_ order printed earlier

X_train_benign = X_train[y_train == BENIGN_LABEL]

print(f"Benign training flows: {X_train_benign.shape}  (took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 3. Build the autoencoder
# architecture: input -> 64 -> 32 (bottleneck) -> 64 -> input
# ----------------------------------------------------------------------
print("\nBuilding autoencoder...")

input_dim = X_train_benign.shape[1]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),   # bottleneck layer
    layers.Dense(64, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")  # output same size as input
])

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# ----------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------
print("\nTraining autoencoder...")
t0 = time.time()

history = autoencoder.fit(
    X_train_benign, X_train_benign,   # input = target (reconstruction task)
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

print(f"Training done  (took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 5. Compute reconstruction error on benign training data
# ----------------------------------------------------------------------
print("\nComputing reconstruction errors...")
t0 = time.time()

reconstructions = autoencoder.predict(X_train_benign, batch_size=512, verbose=0)
mse = np.mean(np.square(X_train_benign - reconstructions), axis=1)

print(f"Mean reconstruction error: {mse.mean():.6f}")
print(f"(took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 6. Set threshold at 95th percentile
# ----------------------------------------------------------------------
threshold = np.percentile(mse, 95)
print(f"\nAnomaly threshold (95th percentile): {threshold:.6f}")

# ----------------------------------------------------------------------
# 7. Save model and threshold
# ----------------------------------------------------------------------
print("\nSaving model and threshold...")
t0 = time.time()

autoencoder.save("models/autoencoder.keras")

with open("models/threshold.json", "w") as f:
    json.dump({"threshold": float(threshold)}, f)

print(f"(took {time.time()-t0:.1f}s)")

print(f"\nTotal time: {time.time()-start:.1f}s")