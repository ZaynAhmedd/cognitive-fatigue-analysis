"""
generate_dataset.py
===================
Cognitive Fatigue & Digital Behaviour Study
Shaik Zain Ahmed Hussain | 22261A0541 | MGIT

Generates the full 1,200-row synthetic dataset that simulates
real-world Google Forms survey responses, and saves it as
data/fatigue-data.csv — the canonical dataset for the project.

Run this FIRST before running clean_and_analyse.py:
    python generate_dataset.py

The dataset simulates responses from students and young professionals
across different age groups, devices, content habits, sleep patterns,
and lifestyle behaviours, with statistically realistic correlations
between digital habits and cognitive outcomes (focus, fatigue,
distraction).
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
os.makedirs("data", exist_ok=True)

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
BREAK_FATIGUE = {
    "Never": 1.8, "Rarely": 0.9, "Sometimes": 0.0,
    "Frequently": -0.7, "Always": -1.2,
}
ACTIVITY_MOD = {
    "Sedentary": 1.2, "Light": 0.5, "Moderate": 0.0,
    "Active": -0.5, "High": -1.0,
}


def clip_round(x, lo=1, hi=10):
    return int(round(min(max(float(x), lo), hi)))


def generate_row(i, rng, base_date):
    screen_time    = round(float(np.clip(rng.normal(5.8, 2.3), 0.5, 14.0)), 1)
    sleep_hours    = round(float(np.clip(rng.normal(6.4, 1.4), 3.0, 10.0)), 1)
    task_switches  = int(np.clip(int(rng.normal(8, 5)), 0, 30))
    work_study_hrs = round(float(np.clip(rng.normal(6.0, 2.0), 1.0, 14.0)), 1)
    social_plat    = int(np.clip(rng.integers(1, 7), 1, 6))
    caffeine       = int(np.clip(rng.integers(0, 6), 0, 5))

    content_type  = rng.choice(CONTENT_TYPES, p=[0.28, 0.17, 0.17, 0.15, 0.09, 0.08, 0.06])
    age_group     = rng.choice(AGE_GROUPS,    p=[0.08, 0.45, 0.28, 0.12, 0.07])
    device        = rng.choice(DEVICES,       p=[0.42, 0.35, 0.08, 0.08, 0.07])
    break_freq    = rng.choice(BREAK_FREQ,    p=[0.10, 0.20, 0.35, 0.25, 0.10])
    phys_activity = rng.choice(PHY_ACTIVITY,  p=[0.18, 0.25, 0.30, 0.17, 0.10])
    takes_breaks  = "Yes" if break_freq in ("Sometimes", "Frequently", "Always") else "No"

    nf, nfa, nd = rng.normal(0, 0.9, 3)

    focus = (
        8.2
        - 0.44 * screen_time
        + 0.32 * (sleep_hours - 6.5)
        - 0.08 * task_switches
        + CONTENT_FOCUS_BONUS[content_type]
        + BREAK_FATIGUE[break_freq] * -0.4
        + ACTIVITY_MOD[phys_activity] * -0.3
        - 0.10 * social_plat
        + 0.06 * caffeine
        + nf
    )
    fatigue = (
        2.8
        + 0.35 * screen_time
        - 0.58 * (sleep_hours - 6.5)
        + 0.07 * task_switches
        + BREAK_FATIGUE[break_freq]
        + ACTIVITY_MOD[phys_activity]
        + 0.12 * social_plat
        - 0.05 * caffeine
        + nfa
    )
    distraction = (
        1.8
        + 0.30 * task_switches * CONTENT_DISTRACTION[content_type] / 1.2
        + 0.18 * screen_time
        + BREAK_FATIGUE[break_freq] * 0.5
        + 0.15 * social_plat
        + nd
    )

    submitted_at = (base_date + timedelta(hours=i * 1.8)).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "screen_time":             screen_time,
        "content_type":            content_type,
        "sleep_hours":             sleep_hours,
        "task_switches":           task_switches,
        "takes_breaks":            takes_breaks,
        "focus_rating":            clip_round(focus),
        "fatigue_level":           clip_round(fatigue),
        "distraction_score":       clip_round(distraction),
        "age_group":               age_group,
        "primary_device":          device,
        "work_study_hours":        work_study_hrs,
        "social_media_platforms":  social_plat,
        "break_frequency":         break_freq,
        "caffeine_intake":         caffeine,
        "physical_activity":       phys_activity,
        "submitted_at":            submitted_at,
    }


def main():
    N = 1200
    rng = np.random.default_rng(42)
    base = datetime(2025, 9, 1, 8, 0, 0)

    print(f"Generating {N} synthetic survey responses...")
    rows = [generate_row(i, rng, base) for i in range(N)]
    df = pd.DataFrame(rows)

    out_path = "data/fatigue-data.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[✓] Dataset saved to '{out_path}'")
    print(f"    Rows      : {len(df)}")
    print(f"    Columns   : {len(df.columns)}")
    print(f"    Variables : {list(df.columns)}")
    print(f"\n    Quick stats:")
    print(f"    Avg screen time    : {df['screen_time'].mean():.2f} hrs")
    print(f"    Avg sleep hours    : {df['sleep_hours'].mean():.2f} hrs")
    print(f"    Avg fatigue level  : {df['fatigue_level'].mean():.2f}")
    print(f"    Avg focus rating   : {df['focus_rating'].mean():.2f}")
    print(f"    Avg distraction    : {df['distraction_score'].mean():.2f}")
    print(f"\nContent Type Distribution:")
    print(df["content_type"].value_counts().to_string())
    print(f"\nRun the analysis pipeline: python clean_and_analyse.py")


if __name__ == "__main__":
    main()
