from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_demo_data import generate_demo_data
from scripts.ingest_data import ingest_latest_raw_data
from scripts.run_eda import run_eda
from scripts.run_monitoring import run_monitoring
from scripts.run_scoring import run_scoring
from scripts.run_training import run_training
from src.utils.logging import configure_logging


logger = configure_logging("bootstrap_demo")


def bootstrap_demo():
    generate_demo_data()
    ingest_latest_raw_data()
    run_eda()
    run_training()
    run_scoring()
    run_monitoring()
    logger.info("Demo pipeline bootstrap completed.")


if __name__ == "__main__":
    bootstrap_demo()
