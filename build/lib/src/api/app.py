from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import CustomerRequest
from src.models.explain import explain_customer
from src.models.scoring import score_customers
from src.retention.action_engine import recommend_action
from src.utils.io import load_artifact, load_latest_dataframe
from src.utils.logging import configure_logging


logger = configure_logging("api")
app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")


def resolve_request_frame(request: CustomerRequest) -> pd.DataFrame:
    bundle = load_artifact("champion_model.joblib")
    defaults = bundle["input_defaults"].copy()

    if request.customer_id:
        features = load_latest_dataframe("customer_features")
        row = features.loc[features["customer_id"] == request.customer_id]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"Customer {request.customer_id} not found.")
        return row.copy()

    payload = defaults
    payload.update(request.features or {})
    payload["customer_id"] = request.customer_id or payload.get("customer_id", "adhoc-customer")
    return pd.DataFrame([payload])


def run_scored_request(request: CustomerRequest) -> dict[str, Any]:
    frame = resolve_request_frame(request)
    scored = score_customers(frame, persist=request.persist, batch_id="api_request")
    bundle = load_artifact("champion_model.joblib")
    explainer_bundle = load_artifact("explainer_model.joblib")
    explanation = explain_customer(explainer_bundle, frame.iloc[[0]])
    recommendation = recommend_action(
        frame.iloc[0],
        float(scored.iloc[0]["churn_probability"]),
        explanation=explanation,
        batch_id="api_request",
    )
    return {
        "customer_id": str(frame.iloc[0]["customer_id"]),
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "churn_probability": float(scored.iloc[0]["churn_probability"]),
        "risk_segment": scored.iloc[0]["risk_segment"],
        "recommended_action": recommendation,
        "explanation": explanation,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: CustomerRequest) -> dict[str, Any]:
    try:
        result = run_scored_request(request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="Model artifacts are missing. Run training first.") from error
    return {
        "customer_id": result["customer_id"],
        "model_name": result["model_name"],
        "model_version": result["model_version"],
        "churn_probability": result["churn_probability"],
        "risk_segment": result["risk_segment"],
    }


@app.post("/explain")
def explain(request: CustomerRequest) -> dict[str, Any]:
    try:
        result = run_scored_request(request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="Model artifacts are missing. Run training first.") from error
    return {
        "customer_id": result["customer_id"],
        "risk_segment": result["risk_segment"],
        "churn_probability": result["churn_probability"],
        "explanation": result["explanation"],
    }


@app.post("/recommend")
def recommend(request: CustomerRequest) -> dict[str, Any]:
    try:
        result = run_scored_request(request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="Model artifacts are missing. Run training first.") from error
    return {
        "customer_id": result["customer_id"],
        "risk_segment": result["risk_segment"],
        "churn_probability": result["churn_probability"],
        "recommended_action": result["recommended_action"],
    }
