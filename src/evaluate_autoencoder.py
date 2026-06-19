import numpy as np
from tensorflow import keras
import json
import joblib

# load test data
X_test = np.load("processed/X_test.npy")
y_test = np.load("processed/y_test.npy")

# load trained autoencoder and threshold
autoencoder = keras.models.load_model("models/autoencoder.keras")
with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

label_encoder = joblib.load("models/label_encoder.joblib")

# get reconstruction error for every test flow
reconstructions = autoencoder.predict(X_test, batch_size=512, verbose=0)
mse = np.mean(np.square(X_test - reconstructions), axis=1)

# flag as anomaly if error > threshold
flagged = mse > threshold

# check flagging rate per class
print(f"{'Class':<30} {'Count':>8} {'% Flagged as Anomaly':>22}")
for class_id in np.unique(y_test):
    class_name = label_encoder.inverse_transform([class_id])[0]
    mask = (y_test == class_id)
    pct_flagged = flagged[mask].mean() * 100
    print(f"{class_name:<30} {mask.sum():>8} {pct_flagged:>21.1f}%")