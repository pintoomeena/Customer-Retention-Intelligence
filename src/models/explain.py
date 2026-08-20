from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES


RISK_TEMPLATES = {
    "recency_days": lambda value: f"No product activity for {int(round(value))} days is pushing churn risk up.",
    "sessions_last_30d": lambda value: f"Only {int(round(value))} sessions in the last 30 days signals weaker engagement.",
    "feature_adoption_ratio": lambda value: f"Feature adoption is only {value:.0%}, which is below healthy usage.",
    "activity_trend_slope": lambda value: f"Weekly activity is trending down ({value:.2f} slope).",
    "failed_transactions_180d": lambda value: f"{int(round(value))} failed transactions are increasing churn risk.",
    "avg_payment_delay_days": lambda value: f"Average payment delay of {value:.1f} days is a financial churn warning.",
    "tickets_90d": lambda value: f"{int(round(value))} support tickets in 90 days suggests unresolved friction.",
    "avg_resolution_hours": lambda value: f"Average support resolution time of {value:.1f} hours is too slow.",
    "engagement_score": lambda value: f"Engagement score is low at {value:.1f}.",
    "churn_risk_heuristic": lambda value: f"Rule-based heuristic risk is elevated at {value:.2f}.",
}

PROTECTIVE_TEMPLATES = {
    "feature_adoption_ratio": lambda value: f"Feature adoption is healthy at {value:.0%}, which reduces churn risk.",
    "sessions_last_30d": lambda value: f"{int(round(value))} recent sessions indicate steady engagement.",
    "activity_trend_slope": lambda value: f"Weekly activity is improving ({value:.2f} slope).",
    "avg_ticket_csat": lambda value: f"Support satisfaction is strong at {value:.1f}/5.",
    "avg_payment_delay_days": lambda value: f"Payments are broadly on time with only {value:.1f} delayed days on average.",
}


def _aggregate_local_contributions(bundle: dict[str, Any], row: pd.DataFrame) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    coefficients = model.coef_[0]
    feature_names = preprocessor.get_feature_names_out()

    contributions: dict[str, float] = {}
    for transformed_name, value, coefficient in zip(feature_names, dense[0], coefficients, strict=False):
        original = _resolve_original_feature(transformed_name)
        contributions[original] = contributions.get(original, 0.0) + float(value * coefficient)

    return pd.DataFrame(
        [{"feature": feature, "contribution": contribution} for feature, contribution in contributions.items()]
    ).sort_values("contribution", ascending=False)


def _resolve_original_feature(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1)
    raw = transformed_name.replace("cat__", "", 1)
    for column in CATEGORICAL_FEATURES:
        if raw == column or raw.startswith(f"{column}_"):
            return column
    return raw


def explain_customer(bundle: dict[str, Any], row: pd.DataFrame, top_n: int = 3) -> dict[str, Any]:
    contributions = _aggregate_local_contributions(bundle, row)
    original_values = row.iloc[0].to_dict()

    risk = contributions[contributions["contribution"] > 0].head(top_n)
    protective = contributions[contributions["contribution"] < 0].sort_values("contribution").head(top_n)

    risk_factors = [
        {
            "feature": item["feature"],
            "impact": round(float(item["contribution"]), 4),
            "message": RISK_TEMPLATES.get(item["feature"], lambda value: f"{item['feature']} is elevating risk.")(
                original_values.get(item["feature"], 0)
            ),
        }
        for item in risk.to_dict(orient="records")
    ]
    protective_factors = [
        {
            "feature": item["feature"],
            "impact": round(float(item["contribution"]), 4),
            "message": PROTECTIVE_TEMPLATES.get(
                item["feature"], lambda value: f"{item['feature']} is helping retain this customer."
            )(original_values.get(item["feature"], 0))
            ,
        }
        for item in protective.to_dict(orient="records")
    ]

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "global_feature_importance": bundle["global_feature_importance"][:10],
    }
