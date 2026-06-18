import pandas as pd
import numpy as np
import os
import glob

FLOWS_DIR = "data/flows/"
OUTPUT_CSV = "processed/combined_flows.csv"

def merge_and_clean():
    csv_files = glob.glob(os.path.join(FLOWS_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()  # remove whitespace from column names
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {df.shape}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape before cleaning: {combined.shape}")

    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)
    print(f"Combined shape after dropping Inf/NaN: {combined.shape}")

    # Merge "X - Attempted" labels into their base class "X"
    # Example: "DoS Hulk - Attempted" becomes "DoS Hulk"
    combined['Label'] = combined['Label'].str.replace(' - Attempted', '', regex=False)

    print(f"\nClass distribution:\n{combined['Label'].value_counts()}")

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    merge_and_clean()
    
'''What are the "Attempted" labels?
Most attack classes in CIC-IDS2017 require transmission of a payload to be effective. For any flow belonging to a payload-reliant attack class but that doesn't actually contain a payload, Engelen labels it "X - Attempted" where X is the original attack class. Kaggle
In plain English — imagine an FTP brute force attack. The attacker opens a TCP connection to the victim's FTP port but never actually sends login attempts (no payload). The connection happened, but the attack didn't execute. The original dataset just called this "FTP-Patator" which is wrong — it never actually brute-forced anything. Engelen separates these out honestly.
A concrete example: the very first flow of the FTP-Patator attack has source port 52108 and is actually a successful pre-test login by the dataset authors — no brute force occurs in this flow at all. Engelen correctly relabels it as "FTP-Patator - Attempted."  '''

''' There are three categories of "Attempted":
Category 0 — attacker sent zero payload bytes (connection opened, nothing sent)
Category 1 — attack startup/teardown artefacts (handshake flows before/after the real attack)
Category 2 — single isolated attempt that doesn't constitute a real attack pattern '''

""" We will merge them. Here's the reasoning:
For our project, "Attempted" flows are still malicious intent — an attacker opened a connection to our FTP port even if they never sent a payload. From a detection standpoint you want to flag that. Keeping them as separate classes means our model needs to learn 25 classes, many of which are tiny — "FTP-Patator - Attempted" probably has very few rows, making it nearly impossible to train on properly.
Merging is also what Engelen's own benchmarking code does, and it brings us back to the clean 15-class design our project is built around. """

