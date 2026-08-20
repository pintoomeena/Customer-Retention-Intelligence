from __future__ import annotations

import logging
from pathlib import Path

from src.utils.config import ROOT_DIR


def configure_logging(name: str = "churn") -> logging.Logger:
    """Configure a shared application logger."""

    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(Path(log_dir / f"{name}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
