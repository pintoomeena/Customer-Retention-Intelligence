from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.models.registry import load_feature_lookup_frame, load_prediction_frame, resolve_model_source, source_config
from src.monitoring.drift import compute_data_drift, compute_performance_monitoring
from src.utils.db import MonitoringReport, SessionLocal
from src.utils.io import load_artifact, save_json_report, timestamp_slug
from src.utils.logging import configure_logging


logger = configure_logging("run_monitoring")


def _load_monitoring_inputs(source_key: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    bundle = load_artifact(source_config(source_key)["artifact_name"])
    if source_key == "demo":
        features = load_feature_lookup_frame(source_key)
    else:
        holdout_path = source_config(source_key)["feature_paths_resolved"][0]
        features = pd.read_parquet(holdout_path)
        features["customer_id"] = features["customer_id"].astype(str)

    if source_key == "demo":
        predictions = load_prediction_frame(source_key)
        label_column = "actual_churn"
    else:
        predictions = load_prediction_frame(source_key, validation=True)
        label_column = "churn"

    return features, predictions, {
        "bundle": bundle,
        "label_column": label_column,
    }


def run_monitoring(model_source: str | None = None) -> dict:
    batch_id = timestamp_slug()
    source_key = resolve_model_source(model_source)
    features, predictions, metadata = _load_monitoring_inputs(source_key)
    bundle = metadata["bundle"]
    label_column = metadata["label_column"]

    payload = {
        "batch_id": batch_id,
        "model_source": source_key,
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "data_drift": compute_data_drift(
            features,
            bundle.get("training_baseline", {}),
            feature_columns=bundle.get("numeric_features", []),
        ),
        "performance": compute_performance_monitoring(
            predictions,
            threshold=float(bundle.get("decision_threshold", 0.5)),
            label_column=label_column,
        ),
        "population_size": int(len(features)),
    }

    if source_key == "demo" and "target_churn" in features.columns:
        payload["churn_rate"] = round(float(features["target_churn"].mean()), 4)
    elif source_key == "kaggle_cell2cell" and label_column == "churn" and not predictions.empty:
        payload["validation_churn_rate"] = round(float(predictions["churn"].mean()), 4)

    save_json_report(payload, "monitoring_latest")

    session = SessionLocal()
    try:
        session.add(MonitoringReport(report_type="daily_monitoring", payload_json=payload, batch_id=batch_id))
        session.commit()
    finally:
        session.close()
    logger.info("Created monitoring report for batch %s using source %s", batch_id, source_key)
    return payload


if __name__ == "__main__":
    run_monitoring()
