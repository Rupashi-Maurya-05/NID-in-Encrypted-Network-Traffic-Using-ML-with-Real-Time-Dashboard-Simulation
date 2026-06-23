import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
import joblib
import xgboost as xgb

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
# 2. Train XGBoost
#    multi:softprob — outputs per-class probabilities, not just hard labels
#    tree_method=hist — fast histogram-based training, handles large data well
#    early_stopping_rounds=20 — stop if val loss doesn't improve for 20 rounds
# -----------------------------------------------------------------------
print("\nTraining XGBoost...")
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
    eval_set=[(X_val, y_val)],
    verbose=25    # print every 25 rounds
)

print(f"\nBest iteration: {model.best_iteration}")
print(f"Training done  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 3. Save model
# -----------------------------------------------------------------------
print("\nSaving...")
joblib.dump(model, "models/xgboost.joblib")
print(f"Total time: {time.time()-start:.1f}s")