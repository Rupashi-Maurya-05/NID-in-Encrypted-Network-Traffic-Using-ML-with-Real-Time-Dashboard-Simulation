import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

start = time.time()

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
print("Loading data...")
t0 = time.time()

df = pd.read_csv("processed/combined_flows.csv", low_memory=False)

print(f"Loaded shape: {df.shape}  (took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 2. Drop columns we decided not to use
# ----------------------------------------------------------------------
print("\nDropping unused columns...")
t0 = time.time()

cols_to_drop = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port"]
df = df.drop(columns=cols_to_drop)

# also drop duplicate rows found during EDA
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")
print(f"Shape after dropping columns/duplicates: {df.shape}  (took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 3. Separate features (X) and label (y)
# ----------------------------------------------------------------------
X = df.drop(columns=["Label"])
y = df["Label"]

# ----------------------------------------------------------------------
# 4. Stratified train/test split (80/20) on REAL data, before any sampling
# ----------------------------------------------------------------------
print("\nSplitting train/test...")
t0 = time.time()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,          # keeps class proportions same in train and test
    random_state=42
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}  (took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 5. Encode labels (string -> integer)
# ----------------------------------------------------------------------
print("\nEncoding labels...")
t0 = time.time()

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

joblib.dump(label_encoder, "models/label_encoder.joblib")

print("Classes:", list(label_encoder.classes_))
print(f"(took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 6. Sampling plan (TRAIN ONLY) — undersample BENIGN, SMOTE tiny classes
# ----------------------------------------------------------------------
print("\nApplying sampling (train only)...")
t0 = time.time()

# step 6a: undersample BENIGN down to 300,000 rows
# we do this manually by randomly dropping extra BENIGN rows from train
benign_label_num = label_encoder.transform(["BENIGN"])[0]

train_df = X_train.copy()
train_df["Label"] = y_train_enc

benign_rows = train_df[train_df["Label"] == benign_label_num]
other_rows = train_df[train_df["Label"] != benign_label_num]

if len(benign_rows) > 300000:
    benign_rows = benign_rows.sample(n=300000, random_state=42)

train_df = pd.concat([benign_rows, other_rows], ignore_index=True)

print(f"After BENIGN undersampling: {train_df.shape}")

# step 6b: SMOTE on the tiny classes
# target counts per class (only listing the ones we want to INCREASE)
smote_targets = {
    "Heartbleed": 150,
    "Web Attack - Sql Injection": 150,
    "Infiltration": 350,
    "Web Attack - XSS": 2000,
    "Web Attack - Brute Force": 3000,
}

# convert class names to their encoded numbers
smote_targets_enc = {
    label_encoder.transform([name])[0]: target
    for name, target in smote_targets.items()
}

# SMOTE needs a dict of {class_label: desired_count} for ALL classes,
# so we build the full target dict: classes not in our list keep their current count
current_counts = train_df["Label"].value_counts().to_dict()
full_sampling_strategy = current_counts.copy()
full_sampling_strategy.update(smote_targets_enc)

X_train_sampled = train_df.drop(columns=["Label"])
y_train_sampled = train_df["Label"]

# SMOTE's k_neighbors must be less than the smallest class size we're using as input
# Heartbleed/SQL Injection have very few real rows, so we lower k_neighbors to be safe
smote = SMOTE(
    sampling_strategy=full_sampling_strategy,
    k_neighbors=5,
    random_state=42
)

X_train_final, y_train_final = smote.fit_resample(X_train_sampled, y_train_sampled)

print(f"After SMOTE: {X_train_final.shape}")
print(f"(sampling took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 7. Scale features (fit on train only, apply to both)
# ----------------------------------------------------------------------
print("\nScaling features...")
t0 = time.time()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.joblib")

print(f"(took {time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# 8. Save everything for the training scripts
# ----------------------------------------------------------------------
print("\nSaving processed arrays...")
t0 = time.time()

np.save("processed/X_train.npy", X_train_scaled)
np.save("processed/X_test.npy", X_test_scaled)
np.save("processed/y_train.npy", y_train_final)
np.save("processed/y_test.npy", y_test_enc)

print(f"(took {time.time()-t0:.1f}s)")

print(f"\nTotal time: {time.time()-start:.1f}s")
print("Final train class distribution:")
print(pd.Series(y_train_final).map(lambda i: label_encoder.inverse_transform([i])[0]).value_counts())