from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.scoring import score_customers
from src.utils.config import get_settings
from src.utils.io import load_latest_dataframe
from src.utils.logging import configure_logging


logger = configure_logging("run_scoring")


def run_scoring() -> str:
    settings = get_settings()
    features = load_latest_dataframe("customer_features")
    scored = score_customers(features, model_source="demo")
    path = settings.report_dir / "predictions_latest.csv"
    scored.to_csv(path, index=False)
    logger.info("Saved predictions to %s", path)
    return str(path)


if __name__ == "__main__":
    run_scoring()
