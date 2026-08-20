from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import ActivateModelRequest, CustomerRequest
from src.models.registry import (
    active_model_metadata,
    available_sources,
    resolve_model_source,
    set_active_model_source,
    source_available,
)
from src.models.serving import resolve_customer_frame, score_frame
from src.utils.logging import configure_logging


logger = configure_logging("api")
app = FastAPI(title="Customer Churn Prediction API", version="0.2.0")


def resolve_request_frame(request: CustomerRequest, source_key: str) -> pd.DataFrame:
    if request.customer_id:
        try:
            return resolve_customer_frame(request.customer_id, source_key)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    payload = (request.features or {}).copy()
    payload["customer_id"] = str(payload.get("customer_id", request.customer_id or "adhoc-customer"))
    return pd.DataFrame([payload])


def run_scored_request(request: CustomerRequest) -> dict[str, Any]:
    source_key = resolve_model_source(request.model_source)
    frame = resolve_request_frame(request, source_key)
    scored, _actions, metadata = score_frame(
        frame,
        model_source=source_key,
        persist=request.persist,
        batch_id="api_request",
    )
    if scored.empty:
        raise HTTPException(status_code=500, detail="No predictions were produced for the request.")
    row = scored.iloc[0]
    bundle = metadata["bundle"]
    explanation = {
        "risk_factors": row["explanation_json"],
        "global_feature_importance": bundle.get("global_feature_importance", [])[:10],
    }
    recommendation = {
        "action_type": row["action_type"],
        "channel": row["channel"],
        "payload_json": row["payload_json"],
        "rationale": row["rationale"],
        "status": row["status"],
        "experiment_group": row["experiment_group"],
    }
    return {
        "customer_id": str(row["customer_id"]),
        "model_source": source_key,
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "decision_threshold": row["decision_threshold"],
        "churn_probability": float(row["churn_probability"]),
        "predicted_churn": int(row["predicted_churn"]),
        "risk_segment": row["risk_segment"],
        "recommended_action": recommendation,
        "explanation": explanation,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    try:
        metadata = active_model_metadata()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "status": "ready",
        "active_model": metadata,
        "available_sources": available_sources(),
    }


@app.get("/sources")
def list_sources() -> dict[str, Any]:
    available = available_sources()
    active = None
    try:
        active = resolve_model_source("auto")
    except FileNotFoundError:
        active = None
    return {
        "active_source": active,
        "sources": [
            {
                "source_key": source_key,
                "available": source_available(source_key),
            }
            for source_key in ["demo", "kaggle_cell2cell"]
        ],
        "available_sources": available,
    }


@app.post("/sources/activate")
def activate_source(request: ActivateModelRequest) -> dict[str, Any]:
    try:
        set_active_model_source(request.model_source)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "status": "ok",
        "active_source": request.model_source,
    }


@app.get("/model/info")
def model_info(model_source: str = "auto") -> dict[str, Any]:
    try:
        return active_model_metadata(model_source)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/predict")
def predict(request: CustomerRequest) -> dict[str, Any]:
    try:
        result = run_scored_request(request)
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {
        "customer_id": result["customer_id"],
        "model_source": result["model_source"],
        "model_name": result["model_name"],
        "model_version": result["model_version"],
        "decision_threshold": result["decision_threshold"],
        "churn_probability": result["churn_probability"],
        "predicted_churn": result["predicted_churn"],
        "risk_segment": result["risk_segment"],
    }


@app.post("/explain")
def explain(request: CustomerRequest) -> dict[str, Any]:
    try:
        result = run_scored_request(request)
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {
        "customer_id": result["customer_id"],
        "model_source": result["model_source"],
        "risk_segment": result["risk_segment"],
        "churn_probability": result["churn_probability"],
        "explanation": result["explanation"],
    }


@app.post("/recommend")
def recommend(request: CustomerRequest) -> dict[str, Any]:
    try:
        result = run_scored_request(request)
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {
        "customer_id": result["customer_id"],
        "model_source": result["model_source"],
        "risk_segment": result["risk_segment"],
        "churn_probability": result["churn_probability"],
        "recommended_action": result["recommended_action"],
    }
