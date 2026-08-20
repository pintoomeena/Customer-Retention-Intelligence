from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.datasets.kaggle_cell2cell import load_raw_cell2cell, prepare_cell2cell_frames
from src.models.cell2cell import train_and_score_cell2cell
from src.utils.config import get_settings
from src.utils.logging import configure_logging


logger = configure_logging("run_kaggle_cell2cell_pipeline")


def run_pipeline(force_download: bool = False) -> dict:
    settings = get_settings()
    train_raw, holdout_raw = load_raw_cell2cell(force_download=force_download)
    train_prepared, holdout_prepared = prepare_cell2cell_frames(train_raw, holdout_raw, persist=True)

    artifacts, validation_predictions, holdout_predictions = train_and_score_cell2cell(
        train_prepared,
        holdout_prepared,
    )

    validation_path = settings.report_dir / "kaggle_cell2cell_validation_predictions.csv"
    holdout_path = settings.report_dir / "kaggle_cell2cell_holdout_predictions.csv"
    validation_predictions.to_csv(validation_path, index=False)
    holdout_predictions.to_csv(holdout_path, index=False)

    logger.info("Saved validation predictions to %s", validation_path)
    logger.info("Saved holdout predictions to %s", holdout_path)

    return {
        "model_name": artifacts.model_name,
        "model_version": artifacts.model_version,
        "report_path": str(artifacts.report_path),
        "model_artifact_path": str(artifacts.model_artifact_path),
        "validation_predictions_path": str(validation_path),
        "holdout_predictions_path": str(holdout_path),
    }


if __name__ == "__main__":
    run_pipeline()
