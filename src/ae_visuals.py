"""
ATTEMPTED-LABEL VISUALS (hardcoded, no raw CSV re-read needed)
+ AE BOTTLENECK DIAGRAM (conceptual, instant)
+ optional AE LATENT SPACE plot (needs model + data, moderate)
-----------------------------------------------------------------
Run from your project root:
    python generate_visuals_extra.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

sns.set_style("whitegrid")
OUT = "visuals"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Raw 25-label counts, pasted directly from your merge_and_clean.py output.
# Merged totals below match your previously reported 15-class numbers
# exactly (e.g. DoS Hulk 158469+579=159048, DoS GoldenEye 7567+80=7647),
# so this data is confirmed consistent — no need to re-read raw CSVs.
# ---------------------------------------------------------------------------
RAW_COUNTS = {
    "BENIGN": 1657069,
    "PortScan": 159023,
    "DoS Hulk": 158469,
    "DDoS": 95123,
    "DoS GoldenEye": 7567,
    "DoS slowloris": 4001,
    "FTP-Patator": 3973,
    "DoS Slowhttptest - Attempted": 3367,
    "SSH-Patator": 2980,
    "DoS Slowhttptest": 1742,
    "DoS slowloris - Attempted": 1706,
    "Bot - Attempted": 1470,
    "Web Attack - Brute Force - Attempted": 1214,
    "Bot": 738,
    "Web Attack - XSS - Attempted": 652,
    "DoS Hulk - Attempted": 579,
    "Web Attack - Brute Force": 151,
    "DoS GoldenEye - Attempted": 80,
    "Infiltration": 32,
    "Web Attack - XSS": 27,
    "Infiltration - Attempted": 16,
    "Web Attack - Sql Injection": 12,
    "FTP-Patator - Attempted": 11,
    "Heartbleed": 11,
    "SSH-Patator - Attempted": 8,
}


def _merge_to_base(counts: dict) -> pd.Series:
    merged = {}
    for label, count in counts.items():
        base = label.replace(" - Attempted", "")
        merged[base] = merged.get(base, 0) + count
    return pd.Series(merged)


# ---------------------------------------------------------------------------
# 9. Attempted-label breakdown — base class vs its Attempted variant
# (Section 2.3)
# ---------------------------------------------------------------------------
def attempted_label_breakdown():
    attempted = {k: v for k, v in RAW_COUNTS.items() if "Attempted" in k}
    base_names = sorted(set(k.replace(" - Attempted", "") for k in attempted))

    rows = []
    for base in base_names:
        base_only = RAW_COUNTS.get(base, 0)
        attempted_only = RAW_COUNTS.get(f"{base} - Attempted", 0)
        rows.append({"Class": base, "Base (payload delivered)": base_only,
                      "Attempted (no payload)": attempted_only})
    comp = pd.DataFrame(rows).set_index("Class").sort_values("Attempted (no payload)")

    fig, ax = plt.subplots(figsize=(10, 6))
    comp.plot(kind="barh", stacked=True, ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_xlabel("Count")
    ax.set_title("Attempted-Label Flows Relative to Base Attack Class\n(pre-merge, 25-label raw data)")
    ax.legend(loc="lower right")
    save(fig, "09_attempted_label_breakdown.png")


# ---------------------------------------------------------------------------
# 10. Class distribution before (25 labels) vs after (15 labels) merge
# (Section 2.3 / 3.2)
# ---------------------------------------------------------------------------
def class_distribution_merge_comparison():
    raw = pd.Series(RAW_COUNTS)
    merged = _merge_to_base(RAW_COUNTS)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    raw_sorted = raw.sort_values(ascending=True)
    colors_raw = ["#DD8452" if "Attempted" in l else "#4C72B0" for l in raw_sorted.index]
    axes[0].barh(raw_sorted.index, raw_sorted.values, color=colors_raw)
    axes[0].set_xscale("log")
    axes[0].set_title(f"Before Merge ({len(raw)} labels)")
    axes[0].set_xlabel("Count (log scale)")
    axes[0].legend(handles=[
        mpatches.Patch(color="#4C72B0", label="Base label"),
        mpatches.Patch(color="#DD8452", label="Attempted variant"),
    ], loc="lower right", fontsize=8)

    merged_sorted = merged.sort_values(ascending=True)
    axes[1].barh(merged_sorted.index, merged_sorted.values, color="#55A868")
    axes[1].set_xscale("log")
    axes[1].set_title(f"After Merge ({len(merged)} labels)")
    axes[1].set_xlabel("Count (log scale)")

    fig.suptitle("Effect of Merging 'Attempted' Variants into Base Classes", fontsize=13)
    save(fig, "10_class_distribution_merge_comparison.png")


# ---------------------------------------------------------------------------
# BONUS: Sankey-style flow diagram — which of the 25 raw labels flow into
# which of the 15 final classes. Simplified as a two-column dot-and-line
# diagram (true Sankey needs plotly; this keeps you on matplotlib only).
# ---------------------------------------------------------------------------
def attempted_merge_flow_diagram():
    attempted = {k: v for k, v in RAW_COUNTS.items() if "Attempted" in k}
    base_names = sorted(set(k.replace(" - Attempted", "") for k in attempted))

    fig, ax = plt.subplots(figsize=(10, 7))
    left_x, right_x = 0, 1
    left_labels = [f"{b} - Attempted" for b in base_names] + base_names
    left_labels = sorted(set(left_labels))
    left_y = {label: i for i, label in enumerate(left_labels)}
    right_y = {label: i * (len(left_labels) / len(base_names)) for i, label in enumerate(base_names)}

    for label in left_labels:
        base = label.replace(" - Attempted", "")
        y0, y1 = left_y[label], right_y[base]
        is_attempted = "Attempted" in label
        color = "#DD8452" if is_attempted else "#4C72B0"
        ax.plot([left_x, right_x], [y0, y1], color=color, alpha=0.5, linewidth=1.5)
        ax.text(left_x - 0.02, y0, label, ha="right", va="center", fontsize=7)

    for base, y in right_y.items():
        ax.text(right_x + 0.02, y, base, ha="left", va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(-0.6, 1.6)
    ax.axis("off")
    ax.set_title("Label Merge: 25 Raw Labels -> 15 Final Classes\n(orange = Attempted variant folded into base class)")
    save(fig, "10b_attempted_merge_flow_diagram.png")


# ---------------------------------------------------------------------------
# AE BOTTLENECK CONCEPT DIAGRAM (Section 5.1) — instant, no model/data needed
# Illustrates WHY the architecture works: compressing 78 features down to
# 32 forces the network to learn only the structure common to BENIGN
# traffic; reconstruction then fails (high MSE) on flows that don't fit
# that learned structure.
# ---------------------------------------------------------------------------
def bottleneck_concept_diagram():
    layer_sizes = [78, 64, 32, 64, 78]
    layer_names = ["Input\n(78 features)", "Dense\n(64, ReLU)", "Bottleneck\n(32, ReLU)",
                    "Dense\n(64, ReLU)", "Output\n(78, Sigmoid)"]
    x_positions = [0, 1, 2, 3, 4]
    max_height = 3.0
    heights = [s / max(layer_sizes) * max_height for s in layer_sizes]

    fig, ax = plt.subplots(figsize=(12, 6))

    for x, h, name, size in zip(x_positions, heights, layer_names, layer_sizes):
        color = "#C44E52" if size == 32 else "#4C72B0"
        rect = mpatches.FancyBboxPatch((x - 0.18, -h / 2), 0.36, h,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor="black", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, -h / 2 - 0.35, name, ha="center", va="top", fontsize=9)
        ax.text(x, 0, str(size), ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")

    for i in range(len(x_positions) - 1):
        arrow = FancyArrowPatch((x_positions[i] + 0.2, 0), (x_positions[i + 1] - 0.2, 0),
                                 arrowstyle="-|>", mutation_scale=15, color="gray")
        ax.add_patch(arrow)

    ax.annotate("Encoder\n(compresses)", xy=(1, 1.9), ha="center", fontsize=10, color="#4C72B0")
    ax.annotate("Decoder\n(reconstructs)", xy=(3, 1.9), ha="center", fontsize=10, color="#4C72B0")
    ax.annotate("Bottleneck forces the model to learn\nonly BENIGN traffic's core structure —\nanything that doesn't fit reconstructs poorly",
                xy=(2, -2.3), ha="center", fontsize=9, style="italic", color="#C44E52")

    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-3, 2.3)
    ax.axis("off")
    ax.set_title("Autoencoder Architecture: The Bottleneck Mechanism", fontsize=13)
    save(fig, "14_ae_bottleneck_concept.png")


# ---------------------------------------------------------------------------
# OPTIONAL — actual latent space (2D PCA of the real 32-dim bottleneck
# activations), BENIGN vs. attacks. This is the "AE in action" version,
# rather than the conceptual diagram above.
# MODERATE: needs models/autoencoder.keras + processed/X_test.npy/y_test.npy.
# Only runs the encoder half (78->64->32), so faster than a full
# reconstruction pass, but still a real forward pass over ~419k rows —
# expect roughly 30s-1.5min on CPU. Subsampling below keeps the plot fast.
# ---------------------------------------------------------------------------
def bottleneck_latent_space_plot(sample_size=15000):
    from tensorflow import keras
    import joblib
    from sklearn.decomposition import PCA

    model = keras.models.load_model("models/autoencoder.keras")
    label_encoder = joblib.load("models/label_encoder.joblib")
    X_test = np.load("processed/X_test.npy")
    y_test = np.load("processed/y_test.npy")

    # Build an encoder-only sub-model up to the 32-unit bottleneck layer.
    # Assumes bottleneck is the 3rd Dense layer (index may need adjusting
    # to match your exact model.summary() layer order).
    bottleneck_layer = [l for l in model.layers if l.output_shape[-1] == 32][0]
    encoder = keras.Model(inputs=model.input, outputs=bottleneck_layer.output)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sample, y_sample = X_test[idx], y_test[idx]

    print("Running encoder forward pass on sample...")
    latent = encoder.predict(X_sample, batch_size=2048, verbose=1)

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)

    benign_label = label_encoder.transform(["BENIGN"])[0]
    is_benign = y_sample == benign_label

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(latent_2d[is_benign, 0], latent_2d[is_benign, 1],
               s=4, alpha=0.4, color="#55A868", label="BENIGN")
    ax.scatter(latent_2d[~is_benign, 0], latent_2d[~is_benign, 1],
               s=4, alpha=0.5, color="#C44E52", label="Attacks (all classes)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("32-Dim Bottleneck Activations, PCA-Projected to 2D\n(sampled test set)")
    ax.legend()
    save(fig, "15_ae_bottleneck_latent_space.png")


if __name__ == "__main__":
    attempted_label_breakdown()
    class_distribution_merge_comparison()
    attempted_merge_flow_diagram()
    bottleneck_concept_diagram()
    # bottleneck_latent_space_plot()  # uncomment when ready to run the model
    print("\nDone.")