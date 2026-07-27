"""
CONFIDENCE-GATED DECISION LOGIC DIAGRAM (Section 1.1 / Section 7) — instant
--------------------------------------------------------------------------
Run from your project root:
    python generate_decision_logic_diagram.py
Saves to visuals/17_decision_logic_diagram.png

Reusable in both Section 1.1 (architecture overview) and Section 7
(combined pipeline deep-dive) — same underlying logic, just referenced
twice.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "visuals"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def box(ax, xy, w, h, text, color, fontsize=9, text_color="white", fontweight="bold"):
    x, y = xy
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.95)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
             color=text_color, fontweight=fontweight, wrap=True)
    return rect


def arrow(ax, start, end, color="gray", style="-|>", lw=1.5, connectionstyle=None):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                         color=color, linewidth=lw,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)


def decision_logic_diagram():
    fig, ax = plt.subplots(figsize=(13, 13))

    # Row y-positions, generously spaced to avoid label/box collisions
    Y_INPUT = 15.5
    Y_MODEL = 13.2
    Y_CHECK = 10.6
    Y_AND = 7.8
    Y_OUTPUT = 5.0
    Y_LEGEND = 3.2

    # --- Input flow ---
    box(ax, (5, Y_INPUT), 2.8, 0.9, "Incoming Flow", "#333333", fontsize=11)

    # --- Two parallel model paths ---
    box(ax, (2, Y_MODEL), 3.4, 1.2, "Autoencoder\nReconstruction MSE", "#4C72B0")
    box(ax, (8, Y_MODEL), 3.4, 1.2, "XGBoost\nPredicted Class + Confidence", "#4C72B0")

    arrow(ax, (4.1, Y_INPUT - 0.5), (2.5, Y_MODEL + 0.65))
    arrow(ax, (5.9, Y_INPUT - 0.5), (7.5, Y_MODEL + 0.65))

    # --- Two check boxes ---
    box(ax, (2, Y_CHECK), 3.6, 1.3, "MSE > P95\nThreshold?", "#DD8452", fontsize=10)
    box(ax, (8, Y_CHECK), 3.6, 1.3, "Predicted = BENIGN\nAND\nconfidence < 0.90?", "#DD8452", fontsize=9.5)

    arrow(ax, (2, Y_MODEL - 0.65), (2, Y_CHECK + 0.7))
    arrow(ax, (8, Y_MODEL - 0.65), (8, Y_CHECK + 0.7))

    # --- AND gate ---
    box(ax, (5, Y_AND), 2.0, 1.0, "AND", "#C44E52", fontsize=13)

    # arrows from check boxes down into AND gate, with YES labels placed
    # at the arrow midpoint (offset from both box edge and AND box) so
    # they don't collide with either box's own text
    arrow(ax, (2.6, Y_CHECK - 0.7), (4.3, Y_AND + 0.55))
    arrow(ax, (7.4, Y_CHECK - 0.7), (5.7, Y_AND + 0.55))

    ax.text(3.15, (Y_CHECK + Y_AND) / 2 + 0.35, "YES", fontsize=8.5,
            color="#C44E52", ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
    ax.text(6.85, (Y_CHECK + Y_AND) / 2 + 0.35, "YES", fontsize=8.5,
            color="#C44E52", ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))

    # --- Final outputs ---
    box(ax, (5, Y_OUTPUT), 3.2, 1.2, "Unknown Anomaly", "#C44E52", fontsize=11)
    box(ax, (1.1, Y_OUTPUT), 3.6, 1.2, "XGBoost's predicted\nclass is used", "#55A868", fontsize=9.5)
    box(ax, (8.9, Y_OUTPUT), 3.4, 1.2, "BENIGN\n(AE flag overridden)", "#55A868", fontsize=9.5)

    # center: straight down, "both TRUE" label
    arrow(ax, (5, Y_AND - 0.5), (5, Y_OUTPUT + 0.6))
    ax.text(5.55, (Y_AND + Y_OUTPUT) / 2, "both\nTRUE", fontsize=8, color="#C44E52",
            fontweight="bold", ha="left", va="center")

    # left: curved arrow to "XGBoost's predicted class is used"
    arrow(ax, (4.0, Y_AND - 0.35), (2.2, Y_OUTPUT + 0.6), connectionstyle="arc3,rad=-0.25")
    ax.text(1.3, (Y_AND + Y_OUTPUT) / 2 + 0.5,
            "AE says normal, or\nXGBoost confidently\npredicts non-BENIGN",
            fontsize=7.5, color="#3A7A50", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#55A868", linewidth=0.8))

    # right: curved arrow to "BENIGN (AE flag overridden)"
    arrow(ax, (6.0, Y_AND - 0.35), (7.8, Y_OUTPUT + 0.6), connectionstyle="arc3,rad=0.25")
    ax.text(8.7, (Y_AND + Y_OUTPUT) / 2 + 0.5,
            "AE flags it, but\nXGBoost confidently\npredicts BENIGN",
            fontsize=7.5, color="#3A7A50", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#55A868", linewidth=0.8))

    ax.set_xlim(-0.8, 11.5)
    ax.set_ylim(1.6, 16.3)
    ax.axis("off")
    ax.set_title("Confidence-Gated Decision Logic", fontsize=16, pad=20)

    legend_elements = [
        mpatches.Patch(color="#4C72B0", label="Model output"),
        mpatches.Patch(color="#DD8452", label="Condition check"),
        mpatches.Patch(color="#C44E52", label="Gate / flagged result"),
        mpatches.Patch(color="#55A868", label="Resolved (non-anomaly) result"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10, frameon=False)

    save(fig, "17_decision_logic_diagram.png")


if __name__ == "__main__":
    decision_logic_diagram()
    print("\nDone.")