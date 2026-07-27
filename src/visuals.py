"""
FAST VISUALS
------------
Run this one now. Everything here either uses hardcoded numbers from your
own report/logs, or loads small .npy / .joblib files (not the big raw CSVs).
Expected total runtime: well under a minute, excluding XGBoost predict
on the test set which is typically a few seconds.

Run from your project root (c:\\Users\\AKSHAT\\nids-project):
    python generate_visuals_fast.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set_style("whitegrid")
OUT = "visuals"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Class distribution — log scale (Section 3)
# Hardcoded from your EDA output (already 15-class, post Attempted-merge)
# ---------------------------------------------------------------------------
def class_distribution_log():
    counts = {
        "BENIGN": 1657069, "DoS Hulk": 159048, "PortScan": 159023,
        "DDoS": 95123, "DoS GoldenEye": 7647, "DoS slowloris": 5707,
        "DoS Slowhttptest": 5109, "FTP-Patator": 3984, "SSH-Patator": 2988,
        "Bot": 2208, "Web Attack - Brute Force": 1365, "Web Attack - XSS": 679,
        "Infiltration": 48, "Web Attack - Sql Injection": 12, "Heartbleed": 11,
    }
    s = pd.Series(counts).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(s.index, s.values, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_title("Class Distribution (Log Scale) — CIC-IDS2017, 15 Classes")
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    save(fig, "01_class_distribution_log.png")


# ---------------------------------------------------------------------------
# 2. Before/after sampling — full dataset vs final training set (Section 4.5)
# "Before" = full dataset counts (proxy for pre-sampling scale).
# "After"  = final train distribution after undersample + SMOTE, from your
#            preprocess.py log output.
# NOTE: "Before" here is full-dataset, not train-only pre-sampling — your
# logs didn't print train-only pre-sampling counts per class. Swap in exact
# numbers if you have them; this framing matches the language already in
# your Section 4.5 text (e.g. "BENIGN 1.66M -> 300k").
# ---------------------------------------------------------------------------
def sampling_before_after():
    before = {
        "BENIGN": 1657069, "DoS Hulk": 159048, "PortScan": 159023,
        "DDoS": 95123, "DoS GoldenEye": 7647, "DoS slowloris": 5707,
        "DoS Slowhttptest": 5109, "FTP-Patator": 3984, "SSH-Patator": 2988,
        "Bot": 2208, "Web Attack - Brute Force": 1365, "Web Attack - XSS": 679,
        "Infiltration": 48, "Web Attack - Sql Injection": 12, "Heartbleed": 11,
    }
    after = {
        "BENIGN": 300000, "DoS Hulk": 101790, "PortScan": 101774,
        "DDoS": 60878, "DoS GoldenEye": 4894, "DoS slowloris": 3653,
        "DoS Slowhttptest": 3270, "Web Attack - Brute Force": 3000,
        "FTP-Patator": 2550, "Web Attack - XSS": 2000, "SSH-Patator": 1912,
        "Bot": 1413, "Infiltration": 350, "Heartbleed": 150,
        "Web Attack - Sql Injection": 150,
    }
    classes = list(before.keys())
    df = pd.DataFrame({
        "Before (full dataset)": [before[c] for c in classes],
        "After (final train set)": [after[c] for c in classes],
    }, index=classes)

    fig, ax = plt.subplots(figsize=(11, 6))
    df.plot(kind="bar", ax=ax, color=["#B0B0B0", "#DD8452"])
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Class Distribution: Full Dataset vs. Final Training Set")
    plt.xticks(rotation=45, ha="right")
    save(fig, "02_sampling_before_after.png")


# ---------------------------------------------------------------------------
# 3. Train / Val / Test split diagram (Section 4.4)
# ---------------------------------------------------------------------------
def split_diagram():
    sizes = [1342001, 335501, 419376]
    labels = [f"Train\n{sizes[0]:,} (64%)", f"Val\n{sizes[1]:,} (16%)",
              f"Test\n{sizes[2]:,} (20%)"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, colors=colors, autopct=lambda p: "",
           startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Stratified Train / Validation / Test Split\n"
                  "(Day-based holdout rejected — attack classes concentrated on single days)")
    save(fig, "03_train_val_test_split.png")


# ---------------------------------------------------------------------------
# 4. Autoencoder per-class detection rate (Section 5.3)
# ---------------------------------------------------------------------------
def ae_detection_rate():
    data = {
        "DDoS": 97.6, "DoS Slowhttptest": 99.6, "DoS Hulk": 89.3,
        "DoS slowloris": 87.8, "DoS GoldenEye": 82.1, "Heartbleed": 100.0,
        "PortScan": 49.9, "SSH-Patator": 22.4, "Bot": 2.5,
        "Web Attack - XSS": 2.9, "Web Attack - Brute Force": 3.3,
        "FTP-Patator": 0.0, "Web Attack - Sql Injection": 0.0,
        "BENIGN (false positive rate)": 5.0,
    }
    s = pd.Series(data).sort_values()
    colors = ["#C44E52" if v < 50 else "#55A868" for v in s.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("% of Flows Flagged as Anomaly")
    ax.set_title("Autoencoder Per-Class Detection Rate")
    for i, v in enumerate(s.values):
        ax.text(v + 1, i, f"{v}%", va="center", fontsize=8)
    save(fig, "04_ae_detection_rate.png")


# ---------------------------------------------------------------------------
# 5. XGBoost per-class F1 (Section 6.2)
# ---------------------------------------------------------------------------
def xgboost_f1_bar():
    f1 = {
        "BENIGN": 0.9958, "Bot": 0.6616, "DDoS": 1.0000, "DoS GoldenEye": 0.9971,
        "DoS Hulk": 0.9998, "DoS Slowhttptest": 0.9927, "DoS slowloris": 0.9991,
        "FTP-Patator": 0.9994, "Heartbleed": 0.6667, "Infiltration": 0.8750,
        "PortScan": 0.9654, "SSH-Patator": 1.0000, "Web Attack - Brute Force": 0.7279,
        "Web Attack - Sql Injection": 1.0000, "Web Attack - XSS": 0.4526,
    }
    s = pd.Series(f1).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(s.index, s.values, color="#4C72B0")
    ax.axvline(0.8889, color="red", linestyle="--", linewidth=1, label="Macro Avg F1 (0.8889)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score")
    ax.set_title("XGBoost Per-Class F1 Score (Standalone, 15-Class)")
    ax.legend(loc="lower right")
    save(fig, "05_xgboost_f1_bar.png")


# ---------------------------------------------------------------------------
# 6. Day-based class coverage (Section 11 — PCAP experiment context)
# Standard, well-documented CIC-IDS2017 day/attack mapping.
# ---------------------------------------------------------------------------
def day_coverage_diagram():
    day_classes = {
        "Monday":    ["BENIGN"],
        "Tuesday":   ["FTP-Patator", "SSH-Patator"],
        "Wednesday": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris",
                      "DoS Slowhttptest", "Heartbleed"],
        "Thursday":  ["Web Attack - Brute Force", "Web Attack - XSS",
                      "Web Attack - Sql Injection", "Infiltration"],
        "Friday":    ["Bot", "PortScan", "DDoS"],
    }
    rows = []
    for day, classes in day_classes.items():
        for c in classes:
            rows.append({"Day": day, "Class": c})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 6))
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    y_pos = {c: i for i, c in enumerate(sorted(df["Class"].unique()))}
    colors = {d: c for d, c in zip(days_order, sns.color_palette("Set2", 5))}

    for _, row in df.iterrows():
        ax.barh(y_pos[row["Class"]], 1, left=days_order.index(row["Day"]),
                color=colors[row["Day"]], edgecolor="white")

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_xticks(range(5))
    ax.set_xticklabels(days_order)
    ax.set_title("Attack Class by Day — Why Tue/Thu-only Classes\nVanished from the Second-Half PCAP Test Set")
    save(fig, "06_day_class_coverage.png")


# ---------------------------------------------------------------------------
# 7. Confusion matrix (Section 6) — needs xgboost.joblib + X_test/y_test.npy
# FAST: npy loads instantly, XGBoost predict on ~420k rows is seconds.
# ---------------------------------------------------------------------------
def confusion_matrix_plot():
    from sklearn.metrics import confusion_matrix

    model = joblib.load("models/xgboost.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    X_test = np.load("processed/X_test.npy")
    y_test = np.load("processed/y_test.npy")

    y_pred = model.predict(X_test)
    classes = label_encoder.classes_

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=False, cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax, cbar_kws={"label": "Row-normalized proportion"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("XGBoost Confusion Matrix (Row-Normalized)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    save(fig, "07_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 8. Feature importance (Section 6) — needs xgboost.joblib + column header only
# FAST: reads just the header row of combined_flows.csv (nrows=0), not full file.
# ---------------------------------------------------------------------------
def feature_importance_plot(top_n=20):
    model = joblib.load("models/xgboost.joblib")

    header = pd.read_csv("processed/combined_flows.csv", nrows=0)
    drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port", "Label"]
    feature_names = [c for c in header.columns if c not in drop_cols]

    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(imp_series.index[::-1], imp_series.values[::-1], color="#55A868")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"XGBoost Top {top_n} Feature Importances")
    save(fig, "08_feature_importance.png")


if __name__ == "__main__":
    class_distribution_log()
    sampling_before_after()
    split_diagram()
    ae_detection_rate()
    xgboost_f1_bar()
    day_coverage_diagram()
    confusion_matrix_plot()
    feature_importance_plot()
    print("\nAll fast visuals done.")