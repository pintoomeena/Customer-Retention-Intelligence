from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.eda import build_eda_summary, create_eda_artifacts
from src.utils.io import load_latest_dataframe, save_json_report
from src.utils.logging import configure_logging


logger = configure_logging("run_eda")


def run_eda() -> dict:
    features = load_latest_dataframe("customer_features")
    summary = build_eda_summary(features)
    artifacts = create_eda_artifacts(features)
    payload = {"summary": summary, "artifacts": artifacts}
    save_json_report(payload, "eda_summary")
    logger.info("Created EDA summary and reports")
    return payload


if __name__ == "__main__":
    run_eda()
