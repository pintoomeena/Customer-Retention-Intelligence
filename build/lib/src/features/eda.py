from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.features.engineering import NUMERIC_FEATURES
from src.utils.config import get_settings


def _write_figure(fig, name: str) -> str:
    settings = get_settings()
    path = settings.report_dir / f"{name}.html"
    fig.write_html(path)
    return str(path)


def build_eda_summary(features: pd.DataFrame) -> dict:
    churn_rate = round(float(features["target_churn"].mean()), 4)
    plan_churn = (
        features.groupby("plan")["target_churn"].mean().round(4).sort_values(ascending=False).to_dict()
    )
    segment_counts = features["customer_segment"].value_counts().to_dict()

    numeric_risk = {}
    for column in NUMERIC_FEATURES:
        churn_mean = features.loc[features["target_churn"] == 1, column].mean()
        stay_mean = features.loc[features["target_churn"] == 0, column].mean()
        numeric_risk[column] = round(float(churn_mean - stay_mean), 4)

    top_drivers = dict(
        sorted(numeric_risk.items(), key=lambda item: abs(item[1]), reverse=True)[:8]
    )

    early_warning_signals = [
        signal
        for signal in [
            "Rising recency and a negative activity trend are leading indicators of churn."
            if numeric_risk.get("recency_days", 0) > 0
            else None,
            "Lower feature adoption is strongly associated with churn."
            if numeric_risk.get("feature_adoption_ratio", 0) < 0
            else None,
            "Failed transactions and payment delays precede many churn events."
            if numeric_risk.get("failed_transactions_180d", 0) > 0
            else None,
            "Higher ticket volume with slower resolution is a support-driven churn pattern."
            if numeric_risk.get("avg_resolution_hours", 0) > 0
            else None,
        ]
        if signal
    ]

    return {
        "churn_rate": churn_rate,
        "plan_churn": plan_churn,
        "segment_counts": segment_counts,
        "top_churn_drivers": top_drivers,
        "behavioral_patterns": {
            "dormant_user_churn_rate": round(
                float(
                    features.loc[features["customer_segment"] == "dormant_user", "target_churn"].mean()
                ),
                4,
            ),
            "power_user_churn_rate": round(
                float(features.loc[features["customer_segment"] == "power_user", "target_churn"].mean()),
                4,
            ),
            "avg_engagement_active": round(
                float(features.loc[features["target_churn"] == 0, "engagement_score"].mean()),
                2,
            ),
            "avg_engagement_churned": round(
                float(features.loc[features["target_churn"] == 1, "engagement_score"].mean()),
                2,
            ),
        },
        "early_warning_signals": early_warning_signals,
    }


def create_eda_artifacts(features: pd.DataFrame) -> dict[str, str]:
    cohort = (
        features.groupby(["cohort_month", "plan"])["target_churn"]
        .mean()
        .reset_index()
        .rename(columns={"target_churn": "churn_rate"})
    )
    fig_cohort = px.density_heatmap(
        cohort,
        x="cohort_month",
        y="plan",
        z="churn_rate",
        color_continuous_scale="YlOrRd",
        title="Cohort Churn Rate by Signup Month and Plan",
    )

    trend = (
        features.groupby("cohort_month")
        .agg(churn_rate=("target_churn", "mean"), avg_engagement=("engagement_score", "mean"))
        .reset_index()
    )
    fig_trend = px.line(
        trend,
        x="cohort_month",
        y=["churn_rate", "avg_engagement"],
        markers=True,
        title="Churn and Engagement Trend by Cohort",
    )

    retention = _build_retention_curve(features)
    fig_retention = px.line(
        retention,
        x="tenure_month",
        y="retention_rate",
        color="plan",
        markers=True,
        title="Approximate Retention Curve by Plan",
    )

    usage = (
        features.groupby("customer_segment")[["feature_adoption_ratio", "sessions_last_30d", "target_churn"]]
        .mean()
        .reset_index()
    )
    fig_usage = go.Figure()
    fig_usage.add_trace(
        go.Bar(x=usage["customer_segment"], y=usage["feature_adoption_ratio"], name="Adoption Ratio")
    )
    fig_usage.add_trace(go.Scatter(x=usage["customer_segment"], y=usage["target_churn"], name="Churn Rate"))
    fig_usage.update_layout(title="Feature Adoption vs Churn by Segment")

    return {
        "cohort_heatmap": _write_figure(fig_cohort, "eda_cohort_heatmap"),
        "trend_chart": _write_figure(fig_trend, "eda_trend_chart"),
        "retention_curve": _write_figure(fig_retention, "eda_retention_curve"),
        "usage_vs_churn": _write_figure(fig_usage, "eda_usage_vs_churn"),
    }


def _build_retention_curve(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["tenure_month"] = (frame["tenure_days"] // 30).clip(lower=1)
    grouped = (
        frame.groupby(["plan", "tenure_month"])["target_churn"]
        .mean()
        .reset_index()
        .rename(columns={"target_churn": "churn_rate"})
    )
    grouped["retention_rate"] = 1 - grouped["churn_rate"]
    return grouped
