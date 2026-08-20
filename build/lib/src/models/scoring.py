from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.explain import explain_customer
from src.retention.action_engine import recommend_action, risk_segment
from src.utils.db import PredictionLog, RetentionAction, SessionLocal
from src.utils.io import load_artifact, timestamp_slug
from src.utils.logging import configure_logging


logger = configure_logging("model_scoring")


def prepare_features_for_scoring(features: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    frame = features.copy()
    for column, default in bundle["input_defaults"].items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default)
    return frame[bundle["feature_columns"] + [col for col in frame.columns if col not in bundle["feature_columns"]]]


def score_customers(features: pd.DataFrame, persist: bool = True, batch_id: str | None = None) -> pd.DataFrame:
    batch_id = batch_id or timestamp_slug()
    model_bundle = load_artifact("champion_model.joblib")
    explainer_bundle = load_artifact("explainer_model.joblib")

    frame = prepare_features_for_scoring(features, model_bundle)
    probabilities = model_bundle["pipeline"].predict_proba(frame[model_bundle["feature_columns"]])[:, 1]

    scored_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for idx, probability in enumerate(probabilities):
        original_row = frame.iloc[[idx]].copy()
        explanation = explain_customer(explainer_bundle, original_row)
        action = recommend_action(original_row.iloc[0], probability, explanation=explanation, batch_id=batch_id)
        scored_rows.append(
            {
                "customer_id": str(original_row.iloc[0]["customer_id"]),
                "batch_id": batch_id,
                "model_name": model_bundle["model_name"],
                "model_version": model_bundle["model_version"],
                "churn_probability": round(float(probability), 4),
                "risk_segment": risk_segment(float(probability)),
                "recommended_action": action["action_type"],
                "actual_churn": int(original_row.iloc[0]["target_churn"]) if "target_churn" in original_row.columns else None,
                "explanation_json": explanation["risk_factors"],
            }
        )
        action_rows.append(action)

    predictions_df = pd.DataFrame(scored_rows)
    actions_df = pd.DataFrame(action_rows)

    if persist:
        session = SessionLocal()
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
                    batch_id=batch_id,
                )
            )
        for row in actions_df.to_dict(orient="records"):
            session.add(RetentionAction(**row))
        session.commit()
        session.close()

    logger.info("Scored %s customers for batch %s", len(predictions_df), batch_id)
    return predictions_df.merge(actions_df, on=["customer_id", "batch_id", "churn_probability", "risk_segment"])
