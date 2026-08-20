from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_data_drift(
    current_features: pd.DataFrame,
    baseline: dict[str, dict[str, float]],
    *,
    feature_columns: list[str],
) -> dict[str, Any]:
    drift_rows = []
    for feature in feature_columns:
        if feature not in current_features.columns:
            continue
        numeric = pd.to_numeric(current_features[feature], errors="coerce")
        current_mean = float(numeric.mean()) if numeric.notna().any() else 0.0
        current_missing_rate = float(numeric.isna().mean())
        baseline_stats = baseline.get(feature, {"mean": 0.0, "std": 1.0, "missing_rate": 0.0})
        std = max(float(baseline_stats.get("std", 1.0)), 1e-6)
        standardized_shift = abs(current_mean - float(baseline_stats.get("mean", 0.0))) / std
        missing_rate_shift = abs(current_missing_rate - float(baseline_stats.get("missing_rate", 0.0)))
        drift_rows.append(
            {
                "feature": feature,
                "current_mean": round(current_mean, 4),
                "baseline_mean": baseline_stats.get("mean", 0.0),
                "standardized_shift": round(float(standardized_shift), 4),
                "current_missing_rate": round(current_missing_rate, 4),
                "baseline_missing_rate": baseline_stats.get("missing_rate", 0.0),
                "missing_rate_shift": round(missing_rate_shift, 4),
                "drift_flag": standardized_shift >= 0.5 or missing_rate_shift >= 0.1,
            }
        )
    drift_rows.sort(key=lambda row: (row["drift_flag"], row["standardized_shift"], row["missing_rate_shift"]), reverse=True)
    return {
        "top_drifted_features": drift_rows[:10],
        "drift_flag_count": int(sum(row["drift_flag"] for row in drift_rows)),
    }


def compute_performance_monitoring(
    predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
    label_column: str = "actual_churn",
) -> dict[str, Any]:
    available = predictions.dropna(subset=[label_column])
    if available.empty or available[label_column].nunique() < 2:
        return {
            "samples_with_ground_truth": int(len(available)),
            "metrics_available": False,
            "decision_threshold": round(float(threshold), 4),
        }

    y_true = available[label_column].astype(int)
    y_score = available["churn_probability"].astype(float)
    y_pred = (y_score >= threshold).astype(int)

    ranked = available.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    top_n = max(int(len(ranked) * 0.10), 1)
    top_slice = ranked.head(top_n)
    top_rate = float(top_slice[label_column].mean())
    base_rate = float(available[label_column].mean())
    captured = int(top_slice[label_column].sum())
    total_positives = max(int(available[label_column].sum()), 1)

    return {
        "samples_with_ground_truth": int(len(available)),
        "metrics_available": True,
        "decision_threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_score)), 4),
        "top_decile_positive_rate": round(top_rate, 4),
        "top_decile_lift": round(top_rate / max(base_rate, 1e-6), 4),
        "top_decile_recall": round(captured / total_positives, 4),
    }
