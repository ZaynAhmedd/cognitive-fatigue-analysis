"""
tests/test_pipeline.py
======================
Cognitive Fatigue & Digital Behaviour Study
Shaik Zain Ahmed Hussain | 22261A0541 | MGIT

Unit tests for the analysis pipeline covering data cleaning,
feature engineering, ML model training, and fatigue classification.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal clean DataFrame for unit testing."""
    return pd.DataFrame({
        "screen_time":          [3.0, 6.5, 10.0, 2.0, 8.0],
        "content_type":         ["Study Content", "Short-form Reels/TikTok",
                                 "Social Media Browsing", "Long-form Video", "Messaging"],
        "sleep_hours":          [7.5, 5.5, 4.0, 8.0, 6.0],
        "task_switches":        [2, 12, 18, 1, 9],
        "takes_breaks":         ["Yes", "No", "No", "Yes", "Yes"],
        "focus_rating":         [8, 5, 3, 9, 6],
        "fatigue_level":        [2, 7, 9, 1, 5],
        "distraction_score":    [2, 7, 9, 2, 6],
        "age_group":            ["18-22", "18-22", "22-25", "16-18", "25-30"],
        "primary_device":       ["Laptop", "Smartphone", "Smartphone", "Laptop", "Smartphone"],
        "work_study_hours":     [8.0, 5.0, 3.0, 9.0, 6.0],
        "social_media_platforms":[1, 4, 5, 1, 3],
        "break_frequency":      ["Frequently", "Rarely", "Never", "Always", "Sometimes"],
        "caffeine_intake":      [1, 2, 3, 0, 2],
        "physical_activity":    ["Active", "Light", "Sedentary", "High", "Moderate"],
        "submitted_at":         ["2025-09-01 09:00:00"] * 5,
    })


# ─── Data cleaning tests ──────────────────────────────────────────────

class TestDataCleaning:

    def test_numeric_columns_are_numeric(self, sample_df):
        """All key numeric columns should be numeric dtype."""
        for col in ["screen_time", "sleep_hours", "focus_rating",
                    "fatigue_level", "distraction_score"]:
            assert pd.api.types.is_numeric_dtype(sample_df[col]), \
                f"{col} should be numeric"

    def test_ratings_in_range(self, sample_df):
        """Rating columns should be within [1, 10]."""
        for col in ["focus_rating", "fatigue_level", "distraction_score"]:
            assert sample_df[col].between(1, 10).all(), \
                f"{col} values out of [1, 10] range"

    def test_screen_time_non_negative(self, sample_df):
        assert (sample_df["screen_time"] >= 0).all()

    def test_sleep_hours_non_negative(self, sample_df):
        assert (sample_df["sleep_hours"] >= 0).all()


# ─── Feature engineering tests ───────────────────────────────────────

class TestFeatureEngineering:

    def test_digital_load_index_created(self, sample_df):
        from clean_and_analyse import engineer_features
        df = engineer_features(sample_df.copy())
        assert "digital_load_index" in df.columns

    def test_sleep_deficit_non_negative(self, sample_df):
        from clean_and_analyse import engineer_features
        df = engineer_features(sample_df.copy())
        assert (df["sleep_deficit"] >= 0).all()

    def test_fatigue_risk_score_created(self, sample_df):
        from clean_and_analyse import engineer_features
        df = engineer_features(sample_df.copy())
        assert "fatigue_risk_score" in df.columns

    def test_focus_efficiency_positive(self, sample_df):
        from clean_and_analyse import engineer_features
        df = engineer_features(sample_df.copy())
        assert (df["focus_efficiency"] >= 0).all()

    def test_cognitive_load_score_range(self, sample_df):
        from clean_and_analyse import engineer_features
        df = engineer_features(sample_df.copy())
        # cognitive_load_score is a weighted combination — should be positive
        assert (df["cognitive_load_score"] > 0).all()


# ─── Fatigue classification tests ────────────────────────────────────

class TestFatigueClassification:

    def test_low_risk(self):
        from clean_and_analyse import classify_fatigue
        assert classify_fatigue(1.5)  == "Low Risk"
        assert classify_fatigue(2.99) == "Low Risk"

    def test_moderate_risk(self):
        from clean_and_analyse import classify_fatigue
        assert classify_fatigue(3.0)  == "Moderate Risk"
        assert classify_fatigue(5.99) == "Moderate Risk"

    def test_high_risk(self):
        from clean_and_analyse import classify_fatigue
        assert classify_fatigue(6.0)  == "High Risk"
        assert classify_fatigue(9.5)  == "High Risk"

    def test_boundary_values(self):
        from clean_and_analyse import classify_fatigue
        assert classify_fatigue(0.0) == "Low Risk"
        assert classify_fatigue(3.0) == "Moderate Risk"
        assert classify_fatigue(6.0) == "High Risk"


# ─── Synthetic data generation tests ─────────────────────────────────

class TestDataGeneration:

    def test_generate_1200_rows(self):
        from clean_and_analyse import generate_synthetic_dataset
        df = generate_synthetic_dataset(n=1200, seed=42)
        assert len(df) == 1200

    def test_required_columns_present(self):
        from clean_and_analyse import generate_synthetic_dataset
        df = generate_synthetic_dataset(n=50, seed=1)
        required = [
            "screen_time", "sleep_hours", "task_switches",
            "focus_rating", "fatigue_level", "distraction_score",
            "content_type", "age_group",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_ratings_within_bounds(self):
        from clean_and_analyse import generate_synthetic_dataset
        df = generate_synthetic_dataset(n=500, seed=2)
        for col in ["focus_rating", "fatigue_level", "distraction_score"]:
            assert df[col].between(1, 10).all(), f"{col} out of bounds"

    def test_no_null_values_in_key_cols(self):
        from clean_and_analyse import generate_synthetic_dataset
        df = generate_synthetic_dataset(n=100, seed=3)
        for col in ["screen_time", "sleep_hours", "focus_rating",
                    "fatigue_level", "distraction_score"]:
            assert df[col].notna().all(), f"Nulls found in {col}"


# ─── Utility helper tests ─────────────────────────────────────────────

class TestHelpers:

    def test_classify_fatigue_risk(self):
        from utils.helpers import classify_fatigue_risk
        assert classify_fatigue_risk(2.0) == "Low Risk"
        assert classify_fatigue_risk(4.5) == "Moderate Risk"
        assert classify_fatigue_risk(7.0) == "High Risk"

    def test_pearson_strength(self):
        from utils.helpers import pearson_strength
        assert pearson_strength(0.8)  == "very strong"
        assert pearson_strength(-0.6) == "strong"
        assert pearson_strength(0.35) == "moderate"
        assert pearson_strength(0.15) == "weak"
        assert pearson_strength(0.05) == "negligible"

    def test_validate_row_valid(self):
        from utils.helpers import validate_row
        row = {
            "screen_time": 5.0, "sleep_hours": 7.0, "task_switches": 8,
            "focus_rating": 7, "fatigue_level": 4, "distraction_score": 3,
            "content_type": "Study Content",
        }
        assert validate_row(row) == []

    def test_validate_row_out_of_range(self):
        from utils.helpers import validate_row
        row = {"focus_rating": 11}  # > 10
        errors = validate_row(row)
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
