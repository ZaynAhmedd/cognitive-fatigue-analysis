"""
clean_and_analyse.py
====================
Cognitive Fatigue & Digital Behaviour Study
Industrial Oriented Mini Project (CS653PC)

Author  : Shaik Zain Ahmed Hussain
Roll No : 22261A0541
College : Mahatma Gandhi Institute of Technology (Autonomous), Hyderabad
Guide   : Dr. C.R.K Reddy, Professor, Dept. of CSE

Description:
    End-to-end pipeline for the Cognitive Fatigue & Digital Behaviour Study.
    Performs data loading, cleaning, synthetic expansion, feature engineering,
    exploratory data analysis, statistical analysis, machine learning model
    training & comparison, fatigue risk classification, and automated
    visualisation — exactly as described in the project documentation.

Usage:
    python clean_and_analyse.py

Output files generated:
    outputs/screen_time_vs_focus.png
    outputs/correlation_heatmap.png
    outputs/fatigue_distribution.png
    outputs/fatigue_boxplot.png
    outputs/feature_importance.png
    outputs/sleep_vs_fatigue.png
    outputs/task_switching_vs_distraction.png
    outputs/content_type_focus.png
    outputs/age_group_analysis.png
    outputs/model_comparison.png
    models/fatigue_model.pkl
    data/fatigue-data.csv          (if not present, synthetic data is generated)
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ──────────────────────────────────

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Refined dark academic palette — sophisticated, not generic
PALETTE = {
    "bg":        "#0d1117",
    "surface":   "#161b22",
    "surface2":  "#21262d",
    "border":    "#30363d",
    "accent":    "#58a6ff",
    "accent2":   "#3fb950",
    "accent3":   "#d29922",
    "accent4":   "#f78166",
    "accent5":   "#bc8cff",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
}

CONTENT_COLORS = {
    "Short-form Reels/TikTok": "#f78166",
    "Social Media Browsing":   "#d29922",
    "Messaging":               "#8b949e",
    "Long-form Video":         "#58a6ff",
    "Study Content":           "#3fb950",
    "Online Gaming":           "#bc8cff",
    "News & Articles":         "#79c0ff",
}

BAR_COLORS = [
    "#58a6ff", "#3fb950", "#d29922", "#f78166",
    "#bc8cff", "#79c0ff", "#56d364",
]


def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "axes.grid":         True,
        "grid.color":        PALETTE["border"],
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "legend.facecolor":  PALETTE["surface2"],
        "legend.edgecolor":  PALETTE["border"],
        "legend.labelcolor": PALETTE["text"],
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    12,
        "figure.dpi":        150,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.facecolor": PALETTE["bg"],
    })


def watermark(ax, text="Cognitive Fatigue & Digital Behaviour Study | MGIT 22261A0541"):
    ax.text(
        0.99, 0.01, text,
        transform=ax.transAxes, fontsize=7.5,
        color=PALETTE["muted"], alpha=0.55,
        ha="right", va="bottom", style="italic",
    )


# ──────────────────────── DATA GENERATION ────────────────────────────

CONTENT_TYPES = [
    "Short-form Reels/TikTok",
    "Long-form Video",
    "Study Content",
    "Social Media Browsing",
    "Messaging",
    "Online Gaming",
    "News & Articles",
]
AGE_GROUPS   = ["16-18", "18-22", "22-25", "25-30", "30+"]
DEVICES      = ["Smartphone", "Laptop", "Tablet", "Desktop", "Multiple Devices"]
BREAK_FREQ   = ["Never", "Rarely", "Sometimes", "Frequently", "Always"]
PHY_ACTIVITY = ["Sedentary", "Light", "Moderate", "Active", "High"]

CONTENT_DISTRACTION = {
    "Short-form Reels/TikTok": 1.70,
    "Social Media Browsing":   1.35,
    "Messaging":               1.05,
    "Online Gaming":           0.85,
    "Long-form Video":         0.60,
    "News & Articles":         0.45,
    "Study Content":           0.20,
}
CONTENT_FOCUS_BONUS = {
    "Short-form Reels/TikTok": -1.80,
    "Social Media Browsing":   -1.20,
    "Messaging":               -0.50,
    "Online Gaming":           -0.30,
    "Long-form Video":          0.20,
    "News & Articles":          0.60,
    "Study Content":            1.80,
}
BREAK_FATIGUE_MODIFIER = {
    "Never": 1.8, "Rarely": 0.9, "Sometimes": 0.0,
    "Frequently": -0.7, "Always": -1.2,
}
ACTIVITY_MODIFIER = {
    "Sedentary": 1.2, "Light": 0.5, "Moderate": 0.0,
    "Active": -0.5, "High": -1.0,
}


def _clip(x, lo=1, hi=10):
    return int(round(min(max(float(x), lo), hi)))


def generate_synthetic_dataset(n=1200, seed=42):
    """
    Generate n=1200 realistic synthetic responses using the same
    causal model as the original seed_data.py, extended with 7
    additional demographic and lifestyle variables.
    """
    rng = np.random.default_rng(seed)

    records = []
    from datetime import datetime, timedelta
    base = datetime(2025, 9, 1, 8, 0)

    for i in range(n):
        screen_time     = float(np.clip(rng.normal(5.8, 2.3), 0.5, 14.0))
        sleep_hours     = float(np.clip(rng.normal(6.4, 1.4), 3.0, 10.0))
        task_switches   = int(np.clip(rng.normal(8, 5), 0, 30))
        work_study_hrs  = float(np.clip(rng.normal(6.0, 2.0), 1.0, 14.0))
        social_plat     = int(np.clip(rng.integers(1, 7), 1, 6))
        caffeine        = int(np.clip(rng.integers(0, 6), 0, 5))
        content_type    = rng.choice(CONTENT_TYPES, p=[0.28, 0.17, 0.17, 0.15, 0.09, 0.08, 0.06])
        age_group       = rng.choice(AGE_GROUPS,   p=[0.08, 0.45, 0.28, 0.12, 0.07])
        device          = rng.choice(DEVICES,      p=[0.42, 0.35, 0.08, 0.08, 0.07])
        break_freq      = rng.choice(BREAK_FREQ,   p=[0.10, 0.20, 0.35, 0.25, 0.10])
        phys_activity   = rng.choice(PHY_ACTIVITY, p=[0.18, 0.25, 0.30, 0.17, 0.10])
        takes_breaks    = "Yes" if break_freq in ("Frequently", "Always", "Sometimes") else "No"

        nf, nfa, nd = rng.normal(0, 0.9, 3)

        focus = (
            8.2
            - 0.44 * screen_time
            + 0.32 * (sleep_hours - 6.5)
            - 0.08 * task_switches
            + CONTENT_FOCUS_BONUS[content_type]
            + BREAK_FATIGUE_MODIFIER[break_freq] * -0.4
            + ACTIVITY_MODIFIER[phys_activity] * -0.3
            - 0.10 * social_plat
            + 0.08 * caffeine
            + nf
        )
        fatigue = (
            2.8
            + 0.35 * screen_time
            - 0.58 * (sleep_hours - 6.5)
            + 0.07 * task_switches
            + BREAK_FATIGUE_MODIFIER[break_freq]
            + ACTIVITY_MODIFIER[phys_activity]
            + 0.12 * social_plat
            - 0.05 * caffeine
            + nfa
        )
        distraction = (
            1.8
            + 0.30 * task_switches * CONTENT_DISTRACTION[content_type] / 1.2
            + 0.18 * screen_time
            + BREAK_FATIGUE_MODIFIER[break_freq] * 0.5
            + 0.15 * social_plat
            + nd
        )

        submitted = (base + timedelta(hours=i * 1.8)).strftime("%Y-%m-%d %H:%M:%S")

        records.append({
            "screen_time":          round(screen_time, 1),
            "content_type":         content_type,
            "sleep_hours":          round(sleep_hours, 1),
            "task_switches":        task_switches,
            "takes_breaks":         takes_breaks,
            "focus_rating":         _clip(focus),
            "fatigue_level":        _clip(fatigue),
            "distraction_score":    _clip(distraction),
            "age_group":            age_group,
            "primary_device":       device,
            "work_study_hours":     round(work_study_hrs, 1),
            "social_media_platforms": social_plat,
            "break_frequency":      break_freq,
            "caffeine_intake":      caffeine,
            "physical_activity":    phys_activity,
            "submitted_at":         submitted,
        })

    return pd.DataFrame(records)


# ──────────────────────── LOAD / CLEAN DATA ──────────────────────────

def load_data(csv_path="data/fatigue-data.csv"):
    if os.path.exists(csv_path):
        print(f"[✓] Loading dataset from '{csv_path}'")
        df = pd.read_csv(csv_path)
        print(f"    {len(df)} rows loaded from CSV.")
    else:
        print("[i] fatigue-data.csv not found — generating 1,200-row synthetic dataset.")
        df = generate_synthetic_dataset(n=1200)
        df.to_csv(csv_path, index=False)
        print(f"[✓] Synthetic dataset saved to '{csv_path}'.")

    # If CSV has only the original 93 rows, expand it synthetically
    if len(df) < 300:
        print(f"[i] Only {len(df)} rows found. Expanding to 1,200 rows with synthetic data.")
        synth = generate_synthetic_dataset(n=1200 - len(df), seed=99)
        # Align columns
        for col in synth.columns:
            if col not in df.columns:
                df[col] = synth[col].iloc[0]
        df = pd.concat([df, synth], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"[✓] Expanded dataset ({len(df)} rows) saved.")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── DATA CLEANING ─────────────────────────────────────")
    original_len = len(df)

    # Force numeric columns
    numeric_cols = [
        "screen_time", "sleep_hours", "task_switches",
        "focus_rating", "fatigue_level", "distraction_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional numeric cols
    for col in ["work_study_hours", "social_media_platforms", "caffeine_intake"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # Drop rows missing critical columns
    df.dropna(subset=numeric_cols, inplace=True)

    # Clip to valid ranges
    df["screen_time"]       = df["screen_time"].clip(0, 24)
    df["sleep_hours"]       = df["sleep_hours"].clip(0, 14)
    df["task_switches"]     = df["task_switches"].clip(0, 40)
    df["focus_rating"]      = df["focus_rating"].clip(1, 10)
    df["fatigue_level"]     = df["fatigue_level"].clip(1, 10)
    df["distraction_score"] = df["distraction_score"].clip(1, 10)

    # Fill missing categorical
    if "content_type" in df.columns:
        df["content_type"].fillna("Short-form Reels/TikTok", inplace=True)
    if "takes_breaks" in df.columns:
        df["takes_breaks"].fillna("No", inplace=True)
    if "age_group" in df.columns:
        df["age_group"].fillna("18-22", inplace=True)
    if "primary_device" in df.columns:
        df["primary_device"].fillna("Smartphone", inplace=True)
    if "break_frequency" in df.columns:
        df["break_frequency"].fillna("Sometimes", inplace=True)
    if "physical_activity" in df.columns:
        df["physical_activity"].fillna("Moderate", inplace=True)

    # Remove statistical outliers (Z-score > 3.5)
    z_cols = ["screen_time", "sleep_hours", "focus_rating", "fatigue_level"]
    for col in z_cols:
        z = np.abs(stats.zscore(df[col].dropna()))
        mask = np.abs(stats.zscore(df[col].fillna(df[col].median()))) < 3.5
        df = df[mask]

    cleaned_len = len(df)
    print(f"    Rows before cleaning : {original_len}")
    print(f"    Rows after  cleaning : {cleaned_len}")
    print(f"    Rows removed         : {original_len - cleaned_len}")
    print(f"    Columns              : {list(df.columns)}")
    return df.reset_index(drop=True)


# ──────────────────────── FEATURE ENGINEERING ────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── FEATURE ENGINEERING ───────────────────────────────")

    df["digital_load_index"] = (
        df["screen_time"] * 3
        + df["task_switches"] * 0.5
        + df.get("social_media_platforms", 2) * 1.2
    ).round(2)

    df["sleep_deficit"] = (8.0 - df["sleep_hours"]).clip(0, 8).round(2)

    df["focus_efficiency"] = (
        df["focus_rating"] / (df["screen_time"].clip(0.1, 24))
    ).round(3)

    df["cognitive_load_score"] = (
        df["fatigue_level"] * 0.4
        + df["distraction_score"] * 0.35
        + (10 - df["focus_rating"]) * 0.25
    ).round(2)

    df["fatigue_risk_score"] = (
        df["fatigue_level"] * 0.45
        + df["distraction_score"] * 0.30
        + df["sleep_deficit"] * 0.25
    ).round(2)

    df["screen_sleep_ratio"] = (
        df["screen_time"] / df["sleep_hours"].clip(0.5, 24)
    ).round(3)

    print(f"    New features created: digital_load_index, sleep_deficit,")
    print(f"    focus_efficiency, cognitive_load_score, fatigue_risk_score,")
    print(f"    screen_sleep_ratio")
    return df


# ──────────────────────── EDA ─────────────────────────────────────────

def run_eda(df: pd.DataFrame):
    print("\n── EXPLORATORY DATA ANALYSIS ─────────────────────────")

    numeric_cols = [
        "screen_time", "sleep_hours", "task_switches",
        "focus_rating", "fatigue_level", "distraction_score",
        "digital_load_index", "sleep_deficit", "cognitive_load_score",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    desc = df[numeric_cols].describe().round(3)
    print("\nDescriptive Statistics:")
    print(desc.to_string())

    print("\nCorrelation with Fatigue Level:")
    corr = df[numeric_cols].corr()["fatigue_level"].sort_values(ascending=False)
    print(corr.to_string())

    print("\nCorrelation with Focus Rating:")
    corr2 = df[numeric_cols].corr()["focus_rating"].sort_values(ascending=False)
    print(corr2.to_string())

    if "content_type" in df.columns:
        print("\nMean Focus by Content Type:")
        ct = df.groupby("content_type")[["focus_rating", "fatigue_level", "distraction_score"]].mean().round(2)
        print(ct.sort_values("focus_rating", ascending=False).to_string())

    return desc, corr


# ──────────────────────── VISUALISATIONS ─────────────────────────────

def _save(fig, name):
    path = f"outputs/{name}"
    fig.savefig(path)
    plt.close(fig)
    print(f"    [✓] Saved → {path}")


def plot_screen_time_vs_focus(df):
    apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Screen Time vs Cognitive Outcomes",
        fontsize=16, fontweight="bold", color=PALETTE["text"], y=1.01,
    )

    # Scatter: Screen Time vs Focus
    ax = axes[0]
    sc = ax.scatter(
        df["screen_time"], df["focus_rating"],
        c=df["fatigue_level"], cmap="RdYlGn_r",
        alpha=0.55, s=22, edgecolors="none",
    )
    m, b, r, p, _ = stats.linregress(df["screen_time"], df["focus_rating"])
    x_line = np.linspace(df["screen_time"].min(), df["screen_time"].max(), 200)
    ax.plot(x_line, m * x_line + b, color=PALETTE["accent"], lw=2.0,
            label=f"Trend (r={r:.2f})")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Fatigue Level", color=PALETTE["muted"], fontsize=10)
    cb.ax.yaxis.set_tick_params(color=PALETTE["muted"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=PALETTE["muted"])
    ax.set_xlabel("Daily Screen Time (hrs)")
    ax.set_ylabel("Focus Rating (1-10)")
    ax.set_title("Screen Time vs Focus Rating")
    ax.legend(framealpha=0.4)
    watermark(ax)

    # Bar: Grouped screen time vs outcomes
    ax2 = axes[1]
    bins  = [0, 2, 4, 6, 8, 24]
    labels= ["<2h", "2-4h", "4-6h", "6-8h", ">8h"]
    d = df.copy()
    d["sg"] = pd.cut(d["screen_time"], bins=bins, labels=labels)
    grp = d.groupby("sg", observed=False)[["focus_rating", "fatigue_level", "distraction_score"]].mean()

    x   = np.arange(len(grp))
    w   = 0.26
    ax2.bar(x - w, grp["focus_rating"],      width=w, color=PALETTE["accent2"], label="Focus", alpha=0.88)
    ax2.bar(x,     grp["fatigue_level"],      width=w, color=PALETTE["accent4"], label="Fatigue", alpha=0.88)
    ax2.bar(x + w, grp["distraction_score"], width=w, color=PALETTE["accent3"], label="Distraction", alpha=0.88)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Screen Time Group")
    ax2.set_ylabel("Average Score (1-10)")
    ax2.set_title("Screen Time Groups vs Cognitive Metrics")
    ax2.legend(framealpha=0.4)
    watermark(ax2)

    fig.tight_layout()
    _save(fig, "screen_time_vs_focus.png")


def plot_correlation_heatmap(df):
    apply_dark_style()
    cols = [
        "screen_time", "sleep_hours", "task_switches",
        "work_study_hours", "social_media_platforms", "caffeine_intake",
        "focus_rating", "fatigue_level", "distraction_score",
        "digital_load_index", "sleep_deficit", "cognitive_load_score",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr().round(3)

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.suptitle(
        "Pearson Correlation Matrix — All Variables",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # show full matrix

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, mask=False,
        cmap=cmap, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8.5, "color": PALETTE["text"]},
        linewidths=0.5, linecolor=PALETTE["border"],
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    watermark(ax)
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")


def plot_fatigue_distribution(df):
    apply_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Distribution of Key Cognitive Outcome Variables",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    for ax, col, color, title in zip(
        axes,
        ["fatigue_level", "focus_rating", "distraction_score"],
        [PALETTE["accent4"], PALETTE["accent2"], PALETTE["accent3"]],
        ["Fatigue Level Distribution", "Focus Rating Distribution", "Distraction Score Distribution"],
    ):
        vals = df[col].dropna()
        ax.hist(vals, bins=10, color=color, alpha=0.80, edgecolor=PALETTE["bg"], linewidth=0.7)

        kde_x = np.linspace(vals.min(), vals.max(), 300)
        kde   = stats.gaussian_kde(vals)
        ax2   = ax.twinx()
        ax2.plot(kde_x, kde(kde_x), color="white", lw=2.0, alpha=0.9)
        ax2.set_ylabel("Density", color=PALETTE["muted"], fontsize=9)
        ax2.tick_params(colors=PALETTE["muted"])
        ax2.yaxis.set_tick_params(labelcolor=PALETTE["muted"])

        ax.axvline(vals.mean(),   color="white", lw=1.5, linestyle="--",
                   label=f"Mean={vals.mean():.2f}")
        ax.axvline(vals.median(), color=PALETTE["accent"], lw=1.5, linestyle=":",
                   label=f"Median={vals.median():.2f}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(framealpha=0.4, fontsize=8)
        watermark(ax)

    fig.tight_layout()
    _save(fig, "fatigue_distribution.png")


def plot_fatigue_boxplot(df):
    apply_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Fatigue Level Across Lifestyle & Behavioural Groups",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    # Content type
    ax = axes[0, 0]
    order = (df.groupby("content_type")["fatigue_level"].median()
               .sort_values(ascending=False).index.tolist()
             if "content_type" in df.columns else None)
    if "content_type" in df.columns:
        colors = [CONTENT_COLORS.get(c, PALETTE["accent"]) for c in order]
        bp = ax.boxplot(
            [df[df["content_type"] == c]["fatigue_level"].dropna().values for c in order],
            patch_artist=True, notch=False,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=PALETTE["muted"]),
            capprops=dict(color=PALETTE["muted"]),
            flierprops=dict(marker=".", color=PALETTE["muted"], alpha=0.3, markersize=3),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticklabels(order, rotation=28, ha="right", fontsize=8)
        ax.set_ylabel("Fatigue Level")
        ax.set_title("Fatigue by Content Type")
        watermark(ax)

    # Break frequency
    ax = axes[0, 1]
    if "break_frequency" in df.columns:
        bf_order = ["Never", "Rarely", "Sometimes", "Frequently", "Always"]
        bf_order = [b for b in bf_order if b in df["break_frequency"].values]
        colors2 = [PALETTE["accent4"], PALETTE["accent3"], PALETTE["accent"],
                   PALETTE["accent2"], "#3fb950"][:len(bf_order)]
        bp2 = ax.boxplot(
            [df[df["break_frequency"] == b]["fatigue_level"].dropna().values for b in bf_order],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=PALETTE["muted"]),
            capprops=dict(color=PALETTE["muted"]),
            flierprops=dict(marker=".", color=PALETTE["muted"], alpha=0.3, markersize=3),
        )
        for patch, col in zip(bp2["boxes"], colors2):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_xticklabels(bf_order, fontsize=9)
        ax.set_ylabel("Fatigue Level")
        ax.set_title("Fatigue by Break Frequency")
        watermark(ax)

    # Physical Activity
    ax = axes[1, 0]
    if "physical_activity" in df.columns:
        pa_order = ["Sedentary", "Light", "Moderate", "Active", "High"]
        pa_order = [p for p in pa_order if p in df["physical_activity"].values]
        bp3 = ax.boxplot(
            [df[df["physical_activity"] == p]["fatigue_level"].dropna().values for p in pa_order],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=PALETTE["muted"]),
            capprops=dict(color=PALETTE["muted"]),
            flierprops=dict(marker=".", color=PALETTE["muted"], alpha=0.3, markersize=3),
        )
        act_colors = [PALETTE["accent4"], PALETTE["accent3"], PALETTE["accent"],
                      PALETTE["accent2"], "#56d364"]
        for patch, col in zip(bp3["boxes"], act_colors[:len(pa_order)]):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_xticklabels(pa_order, fontsize=9)
        ax.set_ylabel("Fatigue Level")
        ax.set_title("Fatigue by Physical Activity Level")
        watermark(ax)

    # Age Group
    ax = axes[1, 1]
    if "age_group" in df.columns:
        age_order = ["16-18", "18-22", "22-25", "25-30", "30+"]
        age_order = [a for a in age_order if a in df["age_group"].values]
        bp4 = ax.boxplot(
            [df[df["age_group"] == a]["fatigue_level"].dropna().values for a in age_order],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=PALETTE["muted"]),
            capprops=dict(color=PALETTE["muted"]),
            flierprops=dict(marker=".", color=PALETTE["muted"], alpha=0.3, markersize=3),
        )
        for patch, col in zip(bp4["boxes"], BAR_COLORS[:len(age_order)]):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_xticklabels(age_order, fontsize=9)
        ax.set_ylabel("Fatigue Level")
        ax.set_title("Fatigue by Age Group")
        watermark(ax)

    fig.tight_layout()
    _save(fig, "fatigue_boxplot.png")


def plot_sleep_vs_fatigue(df):
    apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Sleep Duration vs Fatigue & Focus",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    ax = axes[0]
    sc = ax.scatter(df["sleep_hours"], df["fatigue_level"],
                    c=df["focus_rating"], cmap="RdYlGn", alpha=0.5, s=20, edgecolors="none")
    m, b, r, _, _ = stats.linregress(df["sleep_hours"], df["fatigue_level"])
    xl = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 200)
    ax.plot(xl, m * xl + b, color=PALETTE["accent4"], lw=2, label=f"Trend (r={r:.2f})")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Focus Rating", color=PALETTE["muted"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=PALETTE["muted"])
    ax.set_xlabel("Sleep Duration (hrs)")
    ax.set_ylabel("Fatigue Level (1-10)")
    ax.set_title("Sleep Duration vs Fatigue Level")
    ax.legend(framealpha=0.4)
    watermark(ax)

    ax2 = axes[1]
    bins   = [0, 4, 5.5, 6.5, 7.5, 14]
    labels = ["<4h", "4-5.5h", "5.5-6.5h", "6.5-7.5h", ">7.5h"]
    d = df.copy()
    d["slg"] = pd.cut(d["sleep_hours"], bins=bins, labels=labels)
    grp = d.groupby("slg", observed=False)[["fatigue_level", "focus_rating"]].mean()
    x = np.arange(len(grp))
    ax2.bar(x - 0.2, grp["fatigue_level"], 0.38, color=PALETTE["accent4"], label="Fatigue", alpha=0.85)
    ax2.bar(x + 0.2, grp["focus_rating"],  0.38, color=PALETTE["accent2"], label="Focus",   alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel("Sleep Duration Group")
    ax2.set_ylabel("Average Score")
    ax2.set_title("Sleep Groups: Fatigue vs Focus")
    ax2.legend(framealpha=0.4)
    watermark(ax2)

    fig.tight_layout()
    _save(fig, "sleep_vs_fatigue.png")


def plot_task_switching(df):
    apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Task-Switching Frequency vs Distraction & Focus",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    ax = axes[0]
    ax.scatter(df["task_switches"], df["distraction_score"],
               c=df["focus_rating"], cmap="RdYlGn", alpha=0.5, s=20, edgecolors="none")
    m, b, r, _, _ = stats.linregress(df["task_switches"], df["distraction_score"])
    xl = np.linspace(df["task_switches"].min(), df["task_switches"].max(), 200)
    ax.plot(xl, m * xl + b, color=PALETTE["accent3"], lw=2, label=f"Trend (r={r:.2f})")
    ax.set_xlabel("Task Switches per Hour")
    ax.set_ylabel("Distraction Score (1-10)")
    ax.set_title("Task-Switching vs Distraction")
    ax.legend(framealpha=0.4)
    watermark(ax)

    ax2 = axes[1]
    bins   = [-1, 3, 6, 10, 15, 30]
    labels = ["0-3", "4-6", "7-10", "11-15", "15+"]
    d = df.copy()
    d["tsg"] = pd.cut(d["task_switches"], bins=bins, labels=labels)
    grp = d.groupby("tsg", observed=False)[["distraction_score", "focus_rating", "fatigue_level"]].mean()
    x = np.arange(len(grp))
    w = 0.26
    ax2.bar(x - w, grp["distraction_score"], w, color=PALETTE["accent3"], label="Distraction", alpha=0.85)
    ax2.bar(x,     grp["fatigue_level"],      w, color=PALETTE["accent4"], label="Fatigue",     alpha=0.85)
    ax2.bar(x + w, grp["focus_rating"],       w, color=PALETTE["accent2"], label="Focus",       alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel("Task-Switch Group (per hr)")
    ax2.set_ylabel("Average Score")
    ax2.set_title("Task-Switching Groups vs Cognitive Metrics")
    ax2.legend(framealpha=0.4)
    watermark(ax2)

    fig.tight_layout()
    _save(fig, "task_switching_vs_distraction.png")


def plot_content_type_focus(df):
    if "content_type" not in df.columns:
        return
    apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle(
        "Content Type Consumed vs Cognitive Outcomes",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    grp = df.groupby("content_type")[["focus_rating", "fatigue_level", "distraction_score"]].mean().round(2)
    grp = grp.sort_values("focus_rating", ascending=False)
    colors = [CONTENT_COLORS.get(c, PALETTE["accent"]) for c in grp.index]

    ax = axes[0]
    bars = ax.barh(grp.index, grp["focus_rating"], color=colors, alpha=0.85, edgecolor="none")
    for bar, val in zip(bars, grp["focus_rating"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Average Focus Rating (1-10)")
    ax.set_title("Focus Rating by Content Type")
    ax.set_xlim(0, 11)
    watermark(ax)

    ax2 = axes[1]
    x = np.arange(len(grp))
    w = 0.28
    ax2.bar(x - w, grp["fatigue_level"],      w, color=PALETTE["accent4"], label="Fatigue",     alpha=0.85)
    ax2.bar(x,     grp["distraction_score"],  w, color=PALETTE["accent3"], label="Distraction", alpha=0.85)
    ax2.bar(x + w, grp["focus_rating"],       w, color=PALETTE["accent2"], label="Focus",       alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(grp.index, rotation=28, ha="right", fontsize=8.5)
    ax2.set_ylabel("Average Score")
    ax2.set_title("Grouped Metric Comparison by Content Type")
    ax2.legend(framealpha=0.4)
    watermark(ax2)

    fig.tight_layout()
    _save(fig, "content_type_focus.png")


def plot_age_group_analysis(df):
    if "age_group" not in df.columns:
        return
    apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Age Group Analysis — Screen Time & Cognitive Metrics",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    age_order = ["16-18", "18-22", "22-25", "25-30", "30+"]
    age_order = [a for a in age_order if a in df["age_group"].values]
    grp = df[df["age_group"].isin(age_order)].groupby("age_group")[
        ["screen_time", "sleep_hours", "focus_rating", "fatigue_level"]
    ].mean().reindex(age_order).round(2)
    colors = BAR_COLORS[:len(age_order)]

    ax = axes[0]
    bars = ax.bar(age_order, grp["screen_time"], color=colors, alpha=0.85)
    for bar, val in zip(bars, grp["screen_time"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}h", ha="center", fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Average Daily Screen Time (hrs)")
    ax.set_title("Screen Time by Age Group")
    watermark(ax)

    ax2 = axes[1]
    x = np.arange(len(age_order))
    w = 0.28
    ax2.bar(x - w, grp["focus_rating"],  w, color=PALETTE["accent2"], label="Focus",   alpha=0.85)
    ax2.bar(x,     grp["fatigue_level"], w, color=PALETTE["accent4"], label="Fatigue", alpha=0.85)
    ax2.bar(x + w, grp["sleep_hours"],   w, color=PALETTE["accent"],  label="Sleep",   alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(age_order)
    ax2.set_xlabel("Age Group")
    ax2.set_ylabel("Average Score / Hours")
    ax2.set_title("Focus, Fatigue & Sleep by Age Group")
    ax2.legend(framealpha=0.4)
    watermark(ax2)

    fig.tight_layout()
    _save(fig, "age_group_analysis.png")


# ──────────────────────── MACHINE LEARNING ───────────────────────────

ML_FEATURES = [
    "screen_time", "sleep_hours", "task_switches",
    "focus_rating", "distraction_score",
    "digital_load_index", "sleep_deficit", "focus_efficiency",
]

OPTIONAL_ML = ["work_study_hours", "social_media_platforms", "caffeine_intake"]


def prepare_ml_data(df: pd.DataFrame):
    feats = [f for f in ML_FEATURES if f in df.columns]
    feats += [f for f in OPTIONAL_ML if f in df.columns]
    target = "fatigue_risk_score"
    sub = df[feats + [target]].dropna()
    X = sub[feats].values
    y = sub[target].values
    return X, y, feats


def train_models(df: pd.DataFrame):
    print("\n── MACHINE LEARNING ─────────────────────────────────")
    X, y, feats = prepare_ml_data(df)
    print(f"    Feature matrix shape : {X.shape}")
    print(f"    Target variable      : fatigue_risk_score")
    print(f"    Features used        : {feats}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest":        RandomForestRegressor(n_estimators=200, max_depth=10,
                                                       min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Decision Tree":        DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                                           learning_rate=0.08, random_state=42),
        "Linear Regression":    LinearRegression(),
        "Ridge Regression":     Ridge(alpha=1.0),
    }

    results = {}
    print(f"\n    {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV R² (5-fold)':>16}")
    print("    " + "─" * 66)
    best_r2 = -np.inf
    best_model = None
    best_name  = ""

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, mdl in models.items():
        is_tree = name in ("Random Forest", "Decision Tree", "Gradient Boosting")
        Xtr = X_train if is_tree else X_train_sc
        Xte = X_test  if is_tree else X_test_sc
        Xfull = X if is_tree else scaler.fit_transform(X)

        mdl.fit(Xtr, y_train)
        preds = mdl.predict(Xte)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        cv   = cross_val_score(mdl, Xfull, y, cv=kf, scoring="r2")
        cv_mean = cv.mean()

        results[name] = {
            "mae": mae, "rmse": rmse, "r2": r2, "cv_r2": cv_mean,
            "model": mdl, "preds": preds, "y_test": y_test,
            "feats": feats,
        }

        flag = " ◄ BEST" if r2 > best_r2 else ""
        print(f"    {name:<22} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {cv_mean:>14.4f}{flag}")

        if r2 > best_r2:
            best_r2    = r2
            best_model = mdl
            best_name  = name

    print(f"\n    Best model: {best_name} (R²={best_r2:.4f})")

    # Save best model
    model_path = "models/fatigue_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler,
                     "features": feats, "model_name": best_name}, f)
    print(f"    [✓] Model saved → {model_path}")

    return results, best_model, best_name, feats, scaler


def plot_feature_importance(best_model, feats, model_name):
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle(
        f"Feature Importance — {model_name}",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
    else:
        imp = np.abs(best_model.coef_) / (np.abs(best_model.coef_).sum() + 1e-9)

    importance_df = pd.DataFrame({"Feature": feats, "Importance": imp})
    importance_df = importance_df.sort_values("Importance", ascending=True)

    colors = [PALETTE["accent4"] if imp > importance_df["Importance"].median()
              else PALETTE["accent"] for imp in importance_df["Importance"]]
    bars = ax.barh(importance_df["Feature"], importance_df["Importance"],
                   color=colors, alpha=0.85, edgecolor="none")
    for bar, val in zip(bars, importance_df["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=PALETTE["text"])

    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top Predictors of Fatigue Risk Score ({model_name})")
    watermark(ax)
    fig.tight_layout()
    _save(fig, "feature_importance.png")


def plot_model_comparison(results):
    apply_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Machine Learning Model Comparison",
        fontsize=15, fontweight="bold", color=PALETTE["text"],
    )

    names = list(results.keys())
    metrics = {"MAE": "mae", "RMSE": "rmse", "R² Score": "r2"}
    color_map = {"MAE": PALETTE["accent4"], "RMSE": PALETTE["accent3"], "R² Score": PALETTE["accent2"]}

    for ax, (label, key) in zip(axes, metrics.items()):
        vals = [results[n][key] for n in names]
        bars = ax.bar(names, vals, color=color_map[label], alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", fontsize=8.5, color=PALETTE["text"])
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"{label} Comparison")
        watermark(ax)

    fig.tight_layout()
    _save(fig, "model_comparison.png")


# ──────────────────────── FATIGUE CLASSIFICATION ─────────────────────

def classify_fatigue(score):
    if score < 3.0:
        return "Low Risk"
    elif score < 6.0:
        return "Moderate Risk"
    else:
        return "High Risk"


def run_classification(df):
    print("\n── FATIGUE RISK CLASSIFICATION ──────────────────────")
    df["fatigue_category"] = df["fatigue_risk_score"].apply(classify_fatigue)
    vc = df["fatigue_category"].value_counts()
    print("\n    Fatigue Risk Distribution:")
    for cat, cnt in vc.items():
        pct = cnt / len(df) * 100
        print(f"    {cat:<18}: {cnt:>5} participants ({pct:.1f}%)")
    return df


# ──────────────────────── SAMPLE PREDICTION ──────────────────────────

def sample_prediction():
    print("\n── SAMPLE PREDICTION ────────────────────────────────")
    model_path = "models/fatigue_model.pkl"
    if not os.path.exists(model_path):
        print("    [!] Model not found. Run training first.")
        return

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    mdl     = bundle["model"]
    feats   = bundle["features"]
    name    = bundle["model_name"]

    sample_input = {
        "screen_time":          8.0,
        "sleep_hours":          5.5,
        "task_switches":        14,
        "focus_rating":         4,
        "distraction_score":    8,
        "digital_load_index":   8.0 * 3 + 14 * 0.5 + 3 * 1.2,
        "sleep_deficit":        max(0, 8.0 - 5.5),
        "focus_efficiency":     4 / 8.0,
        "work_study_hours":     9.0,
        "social_media_platforms": 4,
        "caffeine_intake":      3,
    }

    sample_df = pd.DataFrame([{f: sample_input.get(f, 0) for f in feats}])
    pred      = mdl.predict(sample_df)[0]
    category  = classify_fatigue(pred)

    print(f"\n    Model used           : {name}")
    print(f"    Input  (screen_time) : {sample_input['screen_time']} hrs")
    print(f"    Input  (sleep_hours) : {sample_input['sleep_hours']} hrs")
    print(f"    Input  (task_switch) : {sample_input['task_switches']}")
    print(f"    Predicted Fatigue Risk Score : {pred:.3f}")
    print(f"    Fatigue Category             : {category}")


# ──────────────────────── MAIN PIPELINE ──────────────────────────────

def main():
    apply_dark_style()
    print("=" * 62)
    print("  COGNITIVE FATIGUE & DIGITAL BEHAVIOUR STUDY")
    print("  Shaik Zain Ahmed Hussain | 22261A0541 | MGIT")
    print("  Industrial Oriented Mini Project (CS653PC)")
    print("=" * 62)

    # 1. Load & clean
    df = load_data("data/fatigue-data.csv")
    df = clean_data(df)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. EDA
    run_eda(df)

    # 4. Visualisations
    print("\n── GENERATING VISUALISATIONS ─────────────────────────")
    plot_screen_time_vs_focus(df)
    plot_correlation_heatmap(df)
    plot_fatigue_distribution(df)
    plot_fatigue_boxplot(df)
    plot_sleep_vs_fatigue(df)
    plot_task_switching(df)
    plot_content_type_focus(df)
    plot_age_group_analysis(df)

    # 5. ML
    results, best_model, best_name, feats, scaler = train_models(df)
    plot_feature_importance(best_model, feats, best_name)
    plot_model_comparison(results)

    # 6. Classification
    df = run_classification(df)

    # 7. Sample prediction
    sample_prediction()

    print("\n" + "=" * 62)
    print("  Project Execution Completed Successfully")
    print("  All charts saved to: outputs/")
    print("  Trained model saved: models/fatigue_model.pkl")
    print("=" * 62)


if __name__ == "__main__":
    main()
