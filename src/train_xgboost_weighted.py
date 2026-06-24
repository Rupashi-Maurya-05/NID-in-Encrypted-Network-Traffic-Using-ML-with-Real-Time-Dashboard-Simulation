import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
import joblib
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

start = time.time()

# -----------------------------------------------------------------------
# 1. Load data
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
# 2. Compute sample weights
#    Each training sample gets a weight inversely proportional to its
#    class frequency — rare classes get higher weight so the model
#    pays more attention to getting them right during training.
#    "balanced" mode does this automatically using sklearn.
# -----------------------------------------------------------------------
print("\nComputing sample weights...")
t0 = time.time()

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# show effective weight per class
import pandas as pd
weight_df = pd.DataFrame({
    "class": label_encoder.inverse_transform(y_train),
    "weight": sample_weights
}).drop_duplicates().sort_values("weight", ascending=False)

print(weight_df.to_string(index=False))
print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 3. Train XGBoost with sample weights
#    Same hyperparameters as before — only difference is sample_weight
#    passed to fit(), which adjusts the loss contribution per sample
# -----------------------------------------------------------------------
print("\nTraining XGBoost (weighted)...")
t0 = time.time()

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    tree_method="hist",
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=25
)

print(f"\nBest iteration: {model.best_iteration}")
print(f"Training done  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 4. Save with different name so we can compare with original
# -----------------------------------------------------------------------
print("\nSaving...")
joblib.dump(model, "models/xgboost_weighted.joblib")
print(f"Total time: {time.time()-start:.1f}s")