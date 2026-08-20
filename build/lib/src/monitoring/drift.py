from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.features.engineering import NUMERIC_FEATURES


def compute_data_drift(current_features: pd.DataFrame, baseline: dict[str, dict[str, float]]) -> dict[str, Any]:
    drift_rows = []
    for feature in NUMERIC_FEATURES:
        current_mean = float(current_features[feature].mean())
        baseline_stats = baseline.get(feature, {"mean": 0.0, "std": 1.0})
        std = max(float(baseline_stats.get("std", 1.0)), 1e-6)
        standardized_shift = abs(current_mean - float(baseline_stats.get("mean", 0.0))) / std
        drift_rows.append(
            {
                "feature": feature,
                "current_mean": round(current_mean, 4),
                "baseline_mean": baseline_stats.get("mean", 0.0),
                "standardized_shift": round(float(standardized_shift), 4),
                "drift_flag": standardized_shift >= 0.5,
            }
        )
    drift_rows.sort(key=lambda row: row["standardized_shift"], reverse=True)
    return {
        "top_drifted_features": drift_rows[:10],
        "drift_flag_count": int(sum(row["drift_flag"] for row in drift_rows)),
    }


def compute_performance_monitoring(predictions: pd.DataFrame) -> dict[str, Any]:
    available = predictions.dropna(subset=["actual_churn"])
    if available.empty or available["actual_churn"].nunique() < 2:
        return {
            "samples_with_ground_truth": int(len(available)),
            "metrics_available": False,
        }

    y_true = available["actual_churn"].astype(int)
    y_score = available["churn_probability"].astype(float)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "samples_with_ground_truth": int(len(available)),
        "metrics_available": True,
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
    }
