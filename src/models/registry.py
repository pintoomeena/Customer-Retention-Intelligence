from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_settings


DEFAULT_SOURCE_METADATA: dict[str, dict[str, Any]] = {
    "demo": {
        "label": "Synthetic demo pipeline",
        "artifact_name": "champion_model.joblib",
        "explainer_artifact_name": "explainer_model.joblib",
        "report_name": "model_leaderboard.json",
        "prediction_report_name": "predictions_latest.csv",
        "validation_report_name": None,
        "feature_paths": ["customer_features_latest.parquet"],
    },
    "kaggle_cell2cell": {
        "label": "Kaggle Cell2Cell pipeline",
        "artifact_name": "kaggle_cell2cell_model.joblib",
        "explainer_artifact_name": None,
        "report_name": "kaggle_cell2cell_model_report.json",
        "prediction_report_name": "kaggle_cell2cell_holdout_predictions.csv",
        "validation_report_name": "kaggle_cell2cell_validation_predictions.csv",
        "feature_paths": [
            "kaggle_cell2cell_holdout_latest.parquet",
            "kaggle_cell2cell_train_latest.parquet",
        ],
    },
}


def registry_path() -> Path:
    settings = get_settings()
    return settings.artifact_dir / "model_registry.json"


def _default_registry() -> dict[str, Any]:
    return {
        "active_source": None,
        "sources": {},
    }


def load_model_registry() -> dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return _default_registry()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.setdefault("active_source", None)
    payload.setdefault("sources", {})
    return payload


def save_model_registry(payload: dict[str, Any]) -> Path:
    path = registry_path()
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def source_config(source_key: str) -> dict[str, Any]:
    settings = get_settings()
    if source_key not in DEFAULT_SOURCE_METADATA:
        raise KeyError(f"Unknown model source '{source_key}'.")
    metadata = DEFAULT_SOURCE_METADATA[source_key]
    return {
        **metadata,
        "source_key": source_key,
        "artifact_path": settings.artifact_dir / metadata["artifact_name"],
        "explainer_artifact_path": (
            settings.artifact_dir / metadata["explainer_artifact_name"]
            if metadata["explainer_artifact_name"]
            else None
        ),
        "report_path": settings.report_dir / metadata["report_name"],
        "prediction_report_path": settings.report_dir / metadata["prediction_report_name"],
        "validation_report_path": (
            settings.report_dir / metadata["validation_report_name"]
            if metadata["validation_report_name"]
            else None
        ),
        "feature_paths_resolved": [settings.processed_data_dir / name for name in metadata["feature_paths"]],
    }


def source_available(source_key: str) -> bool:
    config = source_config(source_key)
    if not config["artifact_path"].exists() or not config["report_path"].exists():
        return False
    if not config["prediction_report_path"].exists():
        return False
    return all(path.exists() for path in config["feature_paths_resolved"])


def available_sources() -> list[str]:
    return [source_key for source_key in DEFAULT_SOURCE_METADATA if source_available(source_key)]


def _report_mtime(source_key: str) -> float:
    config = source_config(source_key)
    return config["report_path"].stat().st_mtime if config["report_path"].exists() else 0.0


def resolve_model_source(preferred_source: str | None = None) -> str:
    registry = load_model_registry()
    available = available_sources()
    if preferred_source and preferred_source not in {"auto", ""}:
        if preferred_source not in available:
            raise FileNotFoundError(f"Model source '{preferred_source}' is not available.")
        return preferred_source

    active_source = registry.get("active_source")
    if active_source in available:
        return active_source

    if not available:
        raise FileNotFoundError("No model sources are available. Run a training pipeline first.")
    return max(available, key=_report_mtime)


def register_model_source(
    source_key: str,
    *,
    model_name: str,
    model_version: str,
    artifact_name: str,
    report_name: str,
    activate: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    registry = load_model_registry()
    sources = registry.setdefault("sources", {})
    config = source_config(source_key)
    sources[source_key] = {
        "label": config["label"],
        "model_name": model_name,
        "model_version": model_version,
        "artifact_name": artifact_name,
        "report_name": report_name,
        **(extra_metadata or {}),
    }
    if activate:
        registry["active_source"] = source_key
    return save_model_registry(registry)


def set_active_model_source(source_key: str) -> Path:
    if not source_available(source_key):
        raise FileNotFoundError(f"Model source '{source_key}' is not available.")
    registry = load_model_registry()
    registry["active_source"] = source_key
    return save_model_registry(registry)


def active_model_metadata(preferred_source: str | None = None) -> dict[str, Any]:
    source_key = resolve_model_source(preferred_source)
    registry = load_model_registry()
    config = source_config(source_key)
    source_payload = registry.get("sources", {}).get(source_key, {})
    return {
        "source_key": source_key,
        "label": config["label"],
        "artifact_path": str(config["artifact_path"]),
        "report_path": str(config["report_path"]),
        "prediction_report_path": str(config["prediction_report_path"]),
        "validation_report_path": (
            str(config["validation_report_path"]) if config["validation_report_path"] else None
        ),
        **source_payload,
    }


def load_feature_lookup_frame(source_key: str) -> pd.DataFrame:
    config = source_config(source_key)
    frames: list[pd.DataFrame] = []
    for path in config["feature_paths_resolved"]:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["customer_id"] = frame["customer_id"].astype(str)
        if "dataset_split" not in frame.columns:
            frame["dataset_split"] = path.stem.replace("_latest", "")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No feature datasets are available for source '{source_key}'.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["customer_id"], keep="first")
    return combined


def load_prediction_frame(source_key: str, *, validation: bool = False) -> pd.DataFrame:
    config = source_config(source_key)
    path = config["validation_report_path"] if validation else config["prediction_report_path"]
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "customer_id" in frame.columns:
        frame["customer_id"] = frame["customer_id"].astype(str)
    return frame
