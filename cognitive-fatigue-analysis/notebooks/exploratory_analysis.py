"""
notebooks/exploratory_analysis.py
==================================
Cognitive Fatigue & Digital Behaviour Study
Shaik Zain Ahmed Hussain | 22261A0541 | MGIT

Supplementary exploratory analysis script.
Intended to be run as an independent module after clean_and_analyse.py
has been executed and data/fatigue-data.csv is present.

Covers:
  - Pairwise scatter plots across all numeric variables
  - Violin plots for distribution comparisons
  - Device usage vs cognitive outcomes
  - Caffeine intake analysis
  - Social media platform count analysis
  - Fatigue risk score distribution

Run:
    python notebooks/exploratory_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("outputs", exist_ok=True)

PALETTE = {
    "bg": "#0d1117", "surface": "#161b22", "surface2": "#21262d",
    "border": "#30363d", "accent": "#58a6ff", "accent2": "#3fb950",
    "accent3": "#d29922", "accent4": "#f78166", "accent5": "#bc8cff",
    "text": "#e6edf3", "muted": "#8b949e",
}

BAR_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f78166", "#bc8cff", "#79c0ff"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["border"], "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"], "axes.grid": True,
        "grid.color": PALETTE["border"], "grid.linewidth": 0.5,
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "text.color": PALETTE["text"], "legend.facecolor": PALETTE["surface2"],
        "legend.edgecolor": PALETTE["border"], "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "savefig.dpi": 180, "savefig.bbox": "tight",
        "savefig.facecolor": PALETTE["bg"],
    })


def load():
    path = "data/fatigue-data.csv"
    if not os.path.exists(path):
        print("[!] data/fatigue-data.csv not found. Run generate_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[✓] Loaded {len(df)} rows from {path}")
    return df


def engineer(df):
    df["digital_load_index"] = (
        df["screen_time"] * 3
        + df["task_switches"] * 0.5
        + df.get("social_media_platforms", pd.Series([2] * len(df))) * 1.2
    ).round(2)
    df["sleep_deficit"]      = (8.0 - df["sleep_hours"]).clip(0, 8).round(2)
    df["fatigue_risk_score"] = (
        df["fatigue_level"] * 0.45
        + df["distraction_score"] * 0.30
        + df["sleep_deficit"] * 0.25
    ).round(2)
    return df


def wmark(ax):
    ax.text(0.99, 0.01, "MGIT | 22261A0541 | Supplementary Analysis",
            transform=ax.transAxes, fontsize=7, color=PALETTE["muted"],
            alpha=0.5, ha="right", va="bottom", style="italic")


def pairplot(df):
    apply_style()
    cols = ["screen_time", "sleep_hours", "task_switches",
            "focus_rating", "fatigue_level", "distraction_score"]
    cols = [c for c in cols if c in df.columns]
    g = sns.pairplot(df[cols].dropna(), diag_kind="kde", plot_kws={"alpha": 0.3, "s": 8},
                     diag_kws={"fill": True})
    g.figure.suptitle("Pairwise Scatter Plot — All Numeric Variables",
                      y=1.01, fontsize=13, color=PALETTE["text"])
    g.figure.set_facecolor(PALETTE["bg"])
    for ax in g.axes.flatten():
        if ax:
            ax.set_facecolor(PALETTE["surface"])
            ax.tick_params(colors=PALETTE["muted"])
    g.savefig("outputs/pairplot.png", dpi=160)
    plt.close()
    print("  [✓] Saved → outputs/pairplot.png")


def violin_plot(df):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Violin Plots — Cognitive Outcome Distributions",
                 fontsize=14, color=PALETTE["text"])
    for ax, col, color in zip(axes,
        ["focus_rating", "fatigue_level", "distraction_score"],
        [PALETTE["accent2"], PALETTE["accent4"], PALETTE["accent3"]]):
        parts = ax.violinplot(df[col].dropna(), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Score (1-10)")
        wmark(ax)
    fig.tight_layout()
    fig.savefig("outputs/violin_distributions.png")
    plt.close()
    print("  [✓] Saved → outputs/violin_distributions.png")


def device_analysis(df):
    if "primary_device" not in df.columns:
        return
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Primary Device vs Cognitive Metrics", fontsize=14, color=PALETTE["text"])

    grp = df.groupby("primary_device")[["focus_rating", "fatigue_level",
                                         "distraction_score"]].mean().round(2)
    grp = grp.sort_values("focus_rating", ascending=False)
    x   = np.arange(len(grp))
    w   = 0.26
    ax  = axes[0]
    ax.bar(x - w, grp["focus_rating"],     w, color=PALETTE["accent2"], label="Focus",       alpha=0.85)
    ax.bar(x,     grp["fatigue_level"],    w, color=PALETTE["accent4"], label="Fatigue",     alpha=0.85)
    ax.bar(x + w, grp["distraction_score"],w, color=PALETTE["accent3"], label="Distraction", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, rotation=15, ha="right")
    ax.set_ylabel("Average Score")
    ax.set_title("Cognitive Metrics by Device")
    ax.legend(framealpha=0.4)
    wmark(ax)

    cnt = df["primary_device"].value_counts()
    colors = BAR_COLORS[:len(cnt)]
    axes[1].pie(cnt.values, labels=cnt.index, colors=colors,
                autopct="%1.1f%%", pctdistance=0.8,
                wedgeprops=dict(edgecolor=PALETTE["border"], linewidth=1.5))
    axes[1].set_title("Device Distribution")
    axes[1].set_facecolor(PALETTE["bg"])
    wmark(axes[1])

    fig.tight_layout()
    fig.savefig("outputs/device_analysis.png")
    plt.close()
    print("  [✓] Saved → outputs/device_analysis.png")


def caffeine_analysis(df):
    if "caffeine_intake" not in df.columns:
        return
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    grp = df.groupby("caffeine_intake")[["focus_rating", "fatigue_level"]].mean().round(2)
    x   = np.arange(len(grp))
    ax.bar(x - 0.2, grp["focus_rating"],  0.38, color=PALETTE["accent2"], label="Focus",   alpha=0.85)
    ax.bar(x + 0.2, grp["fatigue_level"], 0.38, color=PALETTE["accent4"], label="Fatigue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i} cups/day" for i in grp.index])
    ax.set_ylabel("Average Score")
    ax.set_title("Caffeine Intake vs Focus & Fatigue", fontsize=13)
    ax.legend(framealpha=0.4)
    wmark(ax)
    fig.tight_layout()
    fig.savefig("outputs/caffeine_analysis.png")
    plt.close()
    print("  [✓] Saved → outputs/caffeine_analysis.png")


def fatigue_risk_dist(df):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = df["fatigue_risk_score"].dropna()
    ax.hist(vals, bins=30, color=PALETTE["accent4"], alpha=0.8, edgecolor=PALETTE["bg"])
    ax.axvline(3.0, color=PALETTE["accent3"], lw=2, linestyle="--", label="Low/Moderate boundary")
    ax.axvline(6.0, color=PALETTE["accent2"], lw=2, linestyle="--", label="Moderate/High boundary")
    ax.set_xlabel("Fatigue Risk Score")
    ax.set_ylabel("Count")
    ax.set_title("Fatigue Risk Score Distribution", fontsize=13)
    ax.legend(framealpha=0.4)
    wmark(ax)
    fig.tight_layout()
    fig.savefig("outputs/fatigue_risk_distribution.png")
    plt.close()
    print("  [✓] Saved → outputs/fatigue_risk_distribution.png")


def main():
    apply_style()
    print("=" * 55)
    print("  Supplementary Exploratory Analysis")
    print("  Cognitive Fatigue & Digital Behaviour Study")
    print("  Shaik Zain Ahmed Hussain | 22261A0541 | MGIT")
    print("=" * 55)

    df = load()
    df = engineer(df)

    print("\nGenerating supplementary charts...")
    pairplot(df)
    violin_plot(df)
    device_analysis(df)
    caffeine_analysis(df)
    fatigue_risk_dist(df)

    print("\n[✓] All supplementary charts saved to outputs/")


if __name__ == "__main__":
    main()
