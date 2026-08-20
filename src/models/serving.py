from __future__ import annotations

from typing import Any
import warnings

import pandas as pd

from src.models.cell2cell import recommend_action as recommend_cell2cell_action
from src.models.explain import explain_customer
from src.models.registry import load_feature_lookup_frame, resolve_model_source, source_config
from src.retention.action_engine import recommend_action as recommend_demo_action
from src.retention.action_engine import risk_segment
from src.utils.db import PredictionLog, RetentionAction, SessionLocal
from src.utils.io import load_artifact, timestamp_slug

LGBM_WARNING_PATTERN = "X does not have valid feature names, but LGBMClassifier was fitted with feature names"


DEMO_RISK_RULES = {
    "recency_days": {
        "direction": "high",
        "risk": lambda value: f"No product activity for {int(round(value))} days is a strong churn warning.",
        "protective": lambda value: f"Recent activity within {int(round(value))} days helps retention.",
    },
    "sessions_last_30d": {
        "direction": "low",
        "risk": lambda value: f"Only {int(round(value))} sessions in the last 30 days signals weak engagement.",
        "protective": lambda value: f"{int(round(value))} recent sessions indicate healthy usage momentum.",
    },
    "feature_adoption_ratio": {
        "direction": "low",
        "risk": lambda value: f"Feature adoption is only {value:.0%}, leaving the account under-embedded.",
        "protective": lambda value: f"Feature adoption is healthy at {value:.0%}.",
    },
    "activity_trend_slope": {
        "direction": "low",
        "risk": lambda value: f"Weekly activity is trending down ({value:.2f} slope).",
        "protective": lambda value: f"Weekly activity is improving ({value:.2f} slope).",
    },
    "avg_payment_delay_days": {
        "direction": "high",
        "risk": lambda value: f"Average payment delay of {value:.1f} days is a financial churn signal.",
        "protective": lambda value: f"Payments are broadly on time with only {value:.1f} delayed days on average.",
    },
    "tickets_90d": {
        "direction": "high",
        "risk": lambda value: f"{int(round(value))} support tickets in 90 days suggests unresolved friction.",
        "protective": lambda value: f"Low recent support volume at {int(round(value))} tickets reduces churn pressure.",
    },
}

KAGGLE_RISK_RULES = {
    "perc_change_minutes": {
        "direction": "low",
        "risk": lambda value: f"Usage dropped by {abs(value):.0f} minutes versus the prior period.",
        "protective": lambda value: f"Usage is stable or growing with a {value:.0f} minute change.",
    },
    "perc_change_revenues": {
        "direction": "low",
        "risk": lambda value: f"Revenue declined by {abs(value):.0f}, which often precedes churn.",
        "protective": lambda value: f"Revenue trend is stable with a {value:.0f} change.",
    },
    "call_problem_rate": {
        "direction": "high",
        "risk": lambda value: f"Call-quality friction is elevated with a problem rate of {value:.2%}.",
        "protective": lambda value: f"Call quality is stable with only {value:.2%} problematic events.",
    },
    "monthly_margin": {
        "direction": "low",
        "risk": lambda value: f"Monthly margin is thin at {value:.2f}, limiting perceived account value.",
        "protective": lambda value: f"Monthly margin of {value:.2f} supports proactive retention economics.",
    },
    "months_in_service": {
        "direction": "low",
        "risk": lambda value: f"Only {int(round(value))} months of tenure suggests the account is still fragile.",
        "protective": lambda value: f"{int(round(value))} months of tenure usually improves retention odds.",
    },
    "current_equipment_days": {
        "direction": "high",
        "risk": lambda value: f"The current device has been in use for {int(round(value))} days, a common renewal-risk pattern.",
        "protective": lambda value: f"The device age of {int(round(value))} days does not yet show renewal stress.",
    },
    "retention_pressure_score": {
        "direction": "high",
        "risk": lambda value: "The account has already shown retention pressure, including prior save attempts or offer friction.",
        "protective": lambda value: "There is no recorded retention pressure on this account.",
    },
    "revenue_per_minute": {
        "direction": "low",
        "risk": lambda value: f"Revenue per minute is low at {value:.3f}, indicating weaker monetization quality.",
        "protective": lambda value: f"Revenue per minute of {value:.3f} suggests a healthy usage mix.",
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _training_defaults(bundle: dict[str, Any]) -> dict[str, Any]:
    defaults = bundle.get("input_defaults", {}).copy()
    for column in bundle["feature_columns"]:
        defaults.setdefault(column, 0.0)
    return defaults


def prepare_features_for_source(features: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    frame = features.copy()
    defaults = _training_defaults(bundle)
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default)
    return frame


def resolve_customer_frame(customer_id: str, source_key: str) -> pd.DataFrame:
    features = load_feature_lookup_frame(source_key)
    row = features.loc[features["customer_id"] == str(customer_id)]
    if row.empty:
        raise KeyError(f"Customer {customer_id} not found for source '{source_key}'.")
    return row.iloc[[0]].copy()


def _heuristic_explanation(
    row: pd.DataFrame,
    bundle: dict[str, Any],
    templates: dict[str, dict[str, Any]],
    *,
    top_n: int = 3,
) -> dict[str, Any]:
    baseline = bundle.get("training_baseline", {})
    importance_lookup = {item["feature"]: float(item["importance"]) for item in bundle.get("global_feature_importance", [])}
    selected_features = [feature for feature in importance_lookup if feature in templates][:12]
    risk_factors: list[dict[str, Any]] = []
    protective_factors: list[dict[str, Any]] = []
    row_values = row.iloc[0]

    for feature in selected_features:
        rule = templates[feature]
        stats = baseline.get(feature, {})
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), 1e-6)
        value = _safe_float(row_values.get(feature, mean), mean)
        z_score = (value - mean) / std
        direction = rule["direction"]
        is_risk = z_score >= 0.5 if direction == "high" else z_score <= -0.5
        is_protective = z_score <= -0.5 if direction == "high" else z_score >= 0.5
        payload = {
            "feature": feature,
            "impact": round(abs(z_score) * importance_lookup.get(feature, 0.0), 4),
            "message": rule["risk"](value) if is_risk else rule["protective"](value),
        }
        if is_risk:
            risk_factors.append(payload)
        elif is_protective:
            protective_factors.append(payload)

    if not risk_factors:
        for feature in selected_features[:top_n]:
            value = _safe_float(row_values.get(feature, 0.0))
            risk_factors.append(
                {
                    "feature": feature,
                    "impact": round(importance_lookup.get(feature, 0.0), 4),
                    "message": templates[feature]["risk"](value),
                }
            )

    return {
        "risk_factors": risk_factors[:top_n],
        "protective_factors": protective_factors[:top_n],
        "global_feature_importance": bundle.get("global_feature_importance", [])[:10],
        "mode": "heuristic",
    }


def explain_for_source(source_key: str, bundle: dict[str, Any], row: pd.DataFrame) -> dict[str, Any]:
    if source_key == "demo":
        config = source_config(source_key)
        if config["explainer_artifact_path"] and config["explainer_artifact_path"].exists():
            try:
                explainer_bundle = load_artifact(config["explainer_artifact_path"].name)
                return explain_customer(explainer_bundle, row)
            except RuntimeError:
                pass
        return _heuristic_explanation(row, bundle, DEMO_RISK_RULES)
    return _heuristic_explanation(row, bundle, KAGGLE_RISK_RULES)


def recommend_for_source(
    source_key: str,
    customer_row: pd.Series,
    churn_probability: float,
    explanation: dict[str, Any],
    *,
    batch_id: str,
) -> dict[str, Any]:
    if source_key == "demo":
        return recommend_demo_action(customer_row, churn_probability, explanation=explanation, batch_id=batch_id)
    recommendation = recommend_cell2cell_action(customer_row, churn_probability)
    return {
        "customer_id": str(customer_row["customer_id"]),
        "churn_probability": round(float(churn_probability), 4),
        "risk_segment": recommendation["risk_segment"],
        "action_type": recommendation["recommended_action"],
        "channel": "review_queue",
        "payload_json": {"source": "kaggle_cell2cell"},
        "rationale": recommendation["action_rationale"],
        "experiment_group": "not_applicable",
        "status": "proposed",
        "batch_id": batch_id,
    }


def persist_scored_rows(predictions_df: pd.DataFrame, actions_df: pd.DataFrame) -> None:
    session = SessionLocal()
    try:
        for row in predictions_df.to_dict(orient="records"):
            session.add(
                PredictionLog(
                    prediction_time=pd.Timestamp.utcnow().to_pydatetime(),
                    customer_id=row["customer_id"],
                    model_name=row["model_name"],
                    model_version=row["model_version"],
                    churn_probability=row["churn_probability"],
                    risk_segment=row["risk_segment"],
                    recommended_action=row["recommended_action"],
                    explanation_json=row["explanation_json"],
                    actual_churn=row["actual_churn"],
                    batch_id=row["batch_id"],
                )
            )
        for row in actions_df.to_dict(orient="records"):
            session.add(RetentionAction(**row))
        session.commit()
    finally:
        session.close()


def score_frame(
    features: pd.DataFrame,
    *,
    model_source: str | None = None,
    persist: bool = False,
    batch_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_key = resolve_model_source(model_source)
    bundle = load_artifact(source_config(source_key)["artifact_name"])
    batch_id = batch_id or timestamp_slug()
    frame = prepare_features_for_source(features, bundle)
    feature_columns = bundle["feature_columns"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{LGBM_WARNING_PATTERN}.*", category=UserWarning)
        probabilities = bundle["pipeline"].predict_proba(frame[feature_columns])[:, 1]
    decision_threshold = float(bundle.get("decision_threshold", 0.5))

    scored_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for idx, probability in enumerate(probabilities):
        original_row = frame.iloc[[idx]].copy()
        explanation = explain_for_source(source_key, bundle, original_row)
        action = recommend_for_source(
            source_key,
            original_row.iloc[0],
            float(probability),
            explanation,
            batch_id=batch_id,
        )
        scored_rows.append(
            {
                "customer_id": str(original_row.iloc[0]["customer_id"]),
                "batch_id": batch_id,
                "model_source": source_key,
                "model_name": bundle["model_name"],
                "model_version": bundle["model_version"],
                "churn_probability": round(float(probability), 4),
                "predicted_churn": int(float(probability) >= decision_threshold),
                "decision_threshold": round(decision_threshold, 4),
                "risk_segment": action["risk_segment"] if source_key != "demo" else risk_segment(float(probability)),
                "recommended_action": action["action_type"],
                "actual_churn": (
                    int(original_row.iloc[0].get("target_churn"))
                    if "target_churn" in original_row.columns and not pd.isna(original_row.iloc[0].get("target_churn"))
                    else int(original_row.iloc[0].get("churn"))
                    if "churn" in original_row.columns and not pd.isna(original_row.iloc[0].get("churn"))
                    else None
                ),
                "explanation_json": explanation["risk_factors"],
            }
        )
        action_rows.append(action)

    predictions_df = pd.DataFrame(scored_rows)
    actions_df = pd.DataFrame(action_rows)
    if persist and not predictions_df.empty:
        persist_scored_rows(predictions_df, actions_df)
    merged = predictions_df.merge(actions_df, on=["customer_id", "batch_id", "churn_probability", "risk_segment"])
    return merged, actions_df, {"source_key": source_key, "bundle": bundle}
