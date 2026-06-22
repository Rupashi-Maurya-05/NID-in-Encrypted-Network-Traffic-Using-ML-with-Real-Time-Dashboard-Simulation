import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

start = time.time()

# -----------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------
print("Loading data...")
t0 = time.time()

df = pd.read_csv("processed/combined_flows.csv", low_memory=False)
print(f"Loaded: {df.shape}  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 2. Drop exact full-row duplicates only (3,143 found in EDA)
#    Done BEFORE dropping columns so we compare all original fields
# -----------------------------------------------------------------------
print("\nDropping exact duplicates...")
t0 = time.time()

before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} rows  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 3. Drop non-feature columns
#    Flow ID, Src/Dst IP, Timestamp — identifiers, not generalizable
#    Src Port — ephemeral, no signal
#    Keeping: Dst Port, Protocol — legitimate connection metadata
# -----------------------------------------------------------------------
cols_to_drop = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port"]
df = df.drop(columns=cols_to_drop)
print(f"Shape after dropping columns: {df.shape}")

# -----------------------------------------------------------------------
# 4. Separate features and labels
# -----------------------------------------------------------------------
X = df.drop(columns=["Label"])
y = df["Label"]

feature_names = X.columns.tolist()

# -----------------------------------------------------------------------
# 5. Three-way stratified split: 64% train / 16% val / 20% test
#    Done on real unsampled data — test and val are never touched by SMOTE
#    Test set also becomes the dashboard simulation source
# -----------------------------------------------------------------------
print("\nSplitting train / val / test...")
t0 = time.time()

# first split off 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# then split remaining 80% into 80% train / 20% val
# (80% of 80% = 64% of total, 20% of 80% = 16% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.20,
    stratify=y_temp,
    random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 6. Label encoding
#    Fit on train labels only, apply to val and test
# -----------------------------------------------------------------------
print("\nEncoding labels...")
t0 = time.time()

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc   = label_encoder.transform(y_val)
y_test_enc  = label_encoder.transform(y_test)

joblib.dump(label_encoder, "models/label_encoder.joblib")
print("Classes:", list(label_encoder.classes_))
print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 7. Sampling on TRAIN ONLY
#    Undersample BENIGN to 300k (majority class, massive redundancy)
#    SMOTE on tiny minority classes (see decisions log for reasoning)
#    Val and test stay as real, unsampled data
# -----------------------------------------------------------------------
print("\nSampling train set...")
t0 = time.time()

BENIGN_LABEL = label_encoder.transform(["BENIGN"])[0]

train_df = X_train.copy()
train_df["Label"] = y_train_enc

benign_rows = train_df[train_df["Label"] == BENIGN_LABEL]
other_rows  = train_df[train_df["Label"] != BENIGN_LABEL]

if len(benign_rows) > 300000:
    benign_rows = benign_rows.sample(n=300000, random_state=42)

train_df = pd.concat([benign_rows, other_rows], ignore_index=True)
print(f"After BENIGN undersample: {train_df.shape}")

# SMOTE targets — only for classes too small to train on meaningfully
# reasoning: heavy SMOTE on 11/12 sample classes is fiction,
# but 150 gives the model something to draw a boundary with
smote_targets = {
    "Heartbleed":                150,
    "Web Attack - Sql Injection": 150,
    "Infiltration":               350,
    "Web Attack - XSS":          2000,
    "Web Attack - Brute Force":  3000,
}

smote_targets_enc = {
    label_encoder.transform([name])[0]: count
    for name, count in smote_targets.items()
}

# build full strategy dict: classes not listed keep their current count
current_counts = train_df["Label"].value_counts().to_dict()
full_strategy  = {**current_counts, **smote_targets_enc}

X_tr = train_df.drop(columns=["Label"])
y_tr = train_df["Label"]

smote = SMOTE(sampling_strategy=full_strategy, k_neighbors=5, random_state=42)
X_train_final, y_train_final = smote.fit_resample(X_tr, y_tr)

print(f"After SMOTE: {X_train_final.shape}")
print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 8. Scaling
#    Fit MinMaxScaler on sampled train only
#    Apply same fitted scaler to val and test (no fitting on those)
# -----------------------------------------------------------------------
print("\nScaling...")
t0 = time.time()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.joblib")
print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 9. Save numpy arrays for model training scripts
# -----------------------------------------------------------------------
print("\nSaving arrays...")
t0 = time.time()

np.save("processed/X_train.npy", X_train_scaled)
np.save("processed/X_val.npy",   X_val_scaled)
np.save("processed/X_test.npy",  X_test_scaled)
np.save("processed/y_train.npy", y_train_final)
np.save("processed/y_val.npy",   y_val_enc)
np.save("processed/y_test.npy",  y_test_enc)

print(f"({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 10. Save test set as CSV for dashboard simulation
#     This is the "live traffic" source for the dashboard —
#     model has never trained on these specific flows
# -----------------------------------------------------------------------
print("\nSaving dashboard simulation CSV...")
t0 = time.time()

# combine unscaled test features + original labels for readability
test_csv = X_test.copy()
test_csv["Label"] = y_test.values
test_csv.to_csv("processed/dashboard_simulation.csv", index=False)

print(f"Saved {len(test_csv)} rows to processed/dashboard_simulation.csv")
print(f"({time.time()-t0:.1f}s)")

print(f"\nTotal time: {time.time()-start:.1f}s")

print("\nFinal train class distribution:")
decoded = pd.Series(y_train_final).map(
    lambda i: label_encoder.inverse_transform([i])[0]
)
print(decoded.value_counts())