from __future__ import annotations

import pandas as pd

from src.monitoring.drift import compute_data_drift, compute_performance_monitoring


def test_compute_data_drift_reports_missing_rate_and_shift():
    current = pd.DataFrame(
        {
            "feature_a": [10.0, 11.0, None, 13.0],
            "feature_b": [1.0, 1.0, 1.0, 1.0],
        }
    )
    baseline = {
        "feature_a": {"mean": 5.0, "std": 2.0, "missing_rate": 0.0},
        "feature_b": {"mean": 1.0, "std": 1.0, "missing_rate": 0.0},
    }

    report = compute_data_drift(current, baseline, feature_columns=["feature_a", "feature_b"])

    assert report["drift_flag_count"] >= 1
    assert report["top_drifted_features"][0]["feature"] == "feature_a"
    assert "missing_rate_shift" in report["top_drifted_features"][0]


def test_compute_performance_monitoring_includes_business_metrics():
    predictions = pd.DataFrame(
        {
            "actual_churn": [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            "churn_probability": [0.92, 0.15, 0.81, 0.21, 0.77, 0.33, 0.29, 0.73, 0.1, 0.05],
        }
    )

    report = compute_performance_monitoring(predictions, threshold=0.5)

    assert report["metrics_available"] is True
    assert report["decision_threshold"] == 0.5
    assert "brier_score" in report
    assert "top_decile_lift" in report
    assert "top_decile_recall" in report
