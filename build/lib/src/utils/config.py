from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    database_url: str = Field(
        default=f"sqlite:///{(ROOT_DIR / 'data' / 'db' / 'churn.sqlite3').as_posix()}",
        alias="DATABASE_URL",
    )
    artifact_dir: Path = Field(default=ROOT_DIR / "data" / "artifacts", alias="ARTIFACT_DIR")
    processed_data_dir: Path = Field(
        default=ROOT_DIR / "data" / "processed",
        alias="PROCESSED_DATA_DIR",
    )
    report_dir: Path = Field(default=ROOT_DIR / "data" / "reports", alias="REPORT_DIR")
    raw_data_dir: Path = Field(default=ROOT_DIR / "data" / "raw", alias="RAW_DATA_DIR")
    model_random_state: int = Field(default=42, alias="MODEL_RANDOM_STATE")
    train_test_split: float = Field(default=0.2, alias="TRAIN_TEST_SPLIT")
    cv_folds: int = Field(default=5, alias="CV_FOLDS")
    risk_medium_threshold: float = Field(default=0.40, alias="RISK_MEDIUM_THRESHOLD")
    risk_high_threshold: float = Field(default=0.70, alias="RISK_HIGH_THRESHOLD")
    scoring_lookback_days: int = Field(default=90, alias="SCORING_LOOKBACK_DAYS")
    label_lookback_days: int = Field(default=30, alias="LABEL_LOOKBACK_DAYS")

    @property
    def db_path(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", "", 1))
        return ROOT_DIR / "data" / "db" / "churn.sqlite3"

    def ensure_directories(self) -> None:
        for path in (
            self.artifact_dir,
            self.processed_data_dir,
            self.report_dir,
            self.raw_data_dir,
            self.db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
