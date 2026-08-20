from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.training import train_models
from src.utils.io import load_latest_dataframe
from src.utils.logging import configure_logging


logger = configure_logging("run_training")


def run_training():
    features = load_latest_dataframe("customer_features")
    artifacts = train_models(features)
    logger.info("Champion model: %s (%s)", artifacts.champion_model, artifacts.model_version)
    return artifacts


if __name__ == "__main__":
    run_training()
