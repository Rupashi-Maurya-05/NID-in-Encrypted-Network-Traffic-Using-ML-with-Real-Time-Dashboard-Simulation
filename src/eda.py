import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

""" We need to answer a few concrete questions before we can preprocess correctly:
- What are all the columns, and which are metadata vs real features?
- Are there duplicate rows?
- Are there missing or infinite values?
- What are the data types — any columns that look numeric but are stored as text?
- Any columns with zero variance (same value everywhere) — these add nothing to the model and should be dropped.
- Confirm the label column name and class distribution again from the combined file.
"""

# load the cleaned, merged dataset from the preprocess step
df = pd.read_csv("processed/combined_flows.csv", low_memory=False)

print("Shape:", df.shape)

# 1. all column names
print("\n--- Columns ---")
print(df.columns.tolist())

# 2. dataset info
print("\n--- Dataset Info ---")
df.info()

# 3. data types summary
print("\n--- Data Types ---")
print(df.dtypes.value_counts())

# 4. show exactly which columns are string type
print("\n--- String Columns ---")
print(df.select_dtypes(include=["object", "string"]).columns.tolist())

# 5. confirm important networking fields are numeric
print("\n--- Port / Protocol Data Types ---")
print(df[["Src Port", "Dst Port", "Protocol"]].dtypes)

# 6. missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing.sort_values(ascending=False))

print("Total missing values:", df.isnull().sum().sum())

# 7. duplicate rows
print("\n--- Duplicate Rows ---")
duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)

# 8. columns with only one unique value (zero variance)
print("\n--- Zero Variance Columns ---")
nunique = df.nunique()
zero_var_cols = nunique[nunique == 1].index.tolist()

if zero_var_cols:
    print(zero_var_cols)
else:
    print("None")

# 9. infinite values
print("\n--- Infinite Values ---")

numeric_df = df.select_dtypes(include=["number"])
inf_count = np.isinf(numeric_df).sum().sum()

print("Infinite values:", inf_count)

if inf_count > 0:
    inf_cols = np.isinf(numeric_df).sum()
    inf_cols = inf_cols[inf_cols > 0]
    print("\nColumns containing infinite values:")
    print(inf_cols.sort_values(ascending=False))

# 10. label distribution
print("\n--- Label Distribution ---")
print(df["Label"].value_counts())

print("\n--- Label Distribution (%) ---")
print(round(df["Label"].value_counts(normalize=True) * 100, 4))

# 11. unique labels
print("\n--- Unique Labels ---")
print("Number of classes:", df["Label"].nunique())
print(sorted(df["Label"].unique()))

# 12. memory usage
print("\n--- Memory Usage ---")
memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"{memory_mb:.2f} MB")

# 13. cleaning impact analysis
print("\n--- Cleaning Impact ---")

rows_before = len(df)

# replace infinities with NaN for analysis
df_clean = df.replace([np.inf, -np.inf], np.nan)

rows_after_duplicates = len(df_clean.drop_duplicates())
rows_after_full_clean = len(df_clean.drop_duplicates().dropna())

print("Rows before cleaning:", rows_before)
print("Rows after removing duplicates:", rows_after_duplicates)
print("Rows after removing duplicates + missing values:", rows_after_full_clean)

print(
    "Rows removed by duplicate removal:",
    rows_before - rows_after_duplicates
)

print(
    "Rows removed by duplicate + missing value removal:",
    rows_before - rows_after_full_clean
)

# 14. label distribution plot
plt.figure(figsize=(12, 6))

sns.countplot(
    y="Label",
    data=df,
    order=df["Label"].value_counts().index
)

plt.title("Label Distribution")
plt.xlabel("Count")
plt.ylabel("Label")

plt.tight_layout()
plt.show()