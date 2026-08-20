from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.config import get_settings


def timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_dataframe_version(df: pd.DataFrame, dataset_name: str) -> tuple[Path, str]:
    settings = get_settings()
    version = timestamp_slug()
    path = settings.processed_data_dir / f"{dataset_name}_{version}.parquet"
    df.to_parquet(path, index=False)

    latest_path = settings.processed_data_dir / f"{dataset_name}_latest.parquet"
    df.to_parquet(latest_path, index=False)
    return path, version


def load_latest_dataframe(dataset_name: str) -> pd.DataFrame:
    settings = get_settings()
    path = settings.processed_data_dir / f"{dataset_name}_latest.parquet"
    return pd.read_parquet(path)


def save_json_report(payload: dict[str, Any], name: str) -> Path:
    settings = get_settings()
    path = settings.report_dir / f"{name}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
    return path


def save_artifact(obj: Any, name: str) -> Path:
    settings = get_settings()
    path = settings.artifact_dir / name
    joblib.dump(obj, path)
    return path


def load_artifact(name: str) -> Any:
    settings = get_settings()
    path = settings.artifact_dir / name
    try:
        return joblib.load(path)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load artifact '{name}' from '{path}'. "
            "The artifact was likely generated with a different library version. "
            "Retrain the pipeline in the current environment to regenerate compatible artifacts."
        ) from error
