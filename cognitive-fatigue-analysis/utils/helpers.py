"""
utils/helpers.py
================
Cognitive Fatigue & Digital Behaviour Study
Shaik Zain Ahmed Hussain | 22261A0541 | MGIT

Shared utility functions for data validation, formatting,
and result reporting used across the project pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


def ensure_dirs(*paths):
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def classify_fatigue_risk(score: float) -> str:
    """
    Classify a fatigue risk score into a human-readable category.

    Thresholds align with the documentation:
        Low Risk      : score < 3.0
        Moderate Risk : 3.0 ≤ score < 6.0
        High Risk     : score ≥ 6.0
    """
    if score < 3.0:
        return "Low Risk"
    elif score < 6.0:
        return "Moderate Risk"
    else:
        return "High Risk"


def classify_fatigue_simple(score: float) -> str:
    """Simple Low / Moderate / High classification (from documentation)."""
    if score < 3:
        return "Low"
    elif score < 6:
        return "Moderate"
    else:
        return "High"


def format_percent(value, total):
    """Return 'N (XX.X%)' string."""
    return f"{value} ({value / total * 100:.1f}%)"


def pearson_strength(r: float) -> str:
    """Describe the strength of a Pearson correlation coefficient."""
    a = abs(r)
    if a >= 0.7:
        return "very strong"
    elif a >= 0.5:
        return "strong"
    elif a >= 0.3:
        return "moderate"
    elif a >= 0.1:
        return "weak"
    else:
        return "negligible"


def generate_text_report(df: pd.DataFrame, results: dict, output_path="outputs/report.txt"):
    """
    Generate a plain-text summary report of analysis results,
    suitable for terminal display or saving.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("  COGNITIVE FATIGUE & DIGITAL BEHAVIOUR STUDY")
    lines.append("  Analysis Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("  Shaik Zain Ahmed Hussain | 22261A0541 | MGIT")
    lines.append("=" * 65)
    lines.append("")

    # Dataset summary
    lines.append("DATASET SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total responses  : {len(df)}")
    lines.append(f"  Variables        : {len(df.columns)}")
    lines.append(f"  Date range       : {df['submitted_at'].min() if 'submitted_at' in df.columns else 'N/A'}"
                 f" → {df['submitted_at'].max() if 'submitted_at' in df.columns else 'N/A'}")
    lines.append("")

    # Descriptive stats
    numeric_cols = ["screen_time", "sleep_hours", "task_switches",
                    "focus_rating", "fatigue_level", "distraction_score"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    lines.append("DESCRIPTIVE STATISTICS")
    lines.append("-" * 40)
    desc = df[numeric_cols].describe().round(3)
    lines.append(desc.to_string())
    lines.append("")

    # ML results
    if results:
        lines.append("MACHINE LEARNING RESULTS")
        lines.append("-" * 40)
        lines.append(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
        lines.append("  " + "-" * 50)
        for name, r in results.items():
            lines.append(f"  {name:<22} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>8.4f}")
        lines.append("")

    # Fatigue classification
    if "fatigue_category" in df.columns:
        lines.append("FATIGUE RISK CLASSIFICATION")
        lines.append("-" * 40)
        vc = df["fatigue_category"].value_counts()
        for cat, cnt in vc.items():
            lines.append(f"  {cat:<18}: {format_percent(cnt, len(df))}")
        lines.append("")

    lines.append("=" * 65)

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def validate_row(row: dict) -> list:
    """
    Validate a single survey response row.
    Returns a list of error messages (empty if valid).
    """
    errors = []

    numeric_checks = {
        "screen_time":      (0, 24),
        "sleep_hours":      (0, 14),
        "task_switches":    (0, 40),
        "focus_rating":     (1, 10),
        "fatigue_level":    (1, 10),
        "distraction_score":(1, 10),
    }
    for field, (lo, hi) in numeric_checks.items():
        if field in row:
            try:
                val = float(row[field])
                if not (lo <= val <= hi):
                    errors.append(f"{field} out of range [{lo}, {hi}]: got {val}")
            except (TypeError, ValueError):
                errors.append(f"{field} must be numeric, got: {row[field]!r}")

    valid_content = [
        "Short-form Reels/TikTok", "Long-form Video", "Study Content",
        "Social Media Browsing", "Messaging", "Online Gaming", "News & Articles",
    ]
    if "content_type" in row and row["content_type"] not in valid_content:
        errors.append(f"content_type not recognised: {row['content_type']!r}")

    return errors
