from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import text

import pandas as pd

from src.features.cleaning import (
    add_data_quality_flags,
    clean_activity_events,
    clean_customers,
    clean_support_tickets,
    clean_transaction_events,
)
from src.features.engineering import build_feature_table
from src.utils.config import get_settings
from src.utils.db import DatasetVersion, SessionLocal, get_engine, init_db
from src.utils.io import save_dataframe_version
from src.utils.logging import configure_logging


logger = configure_logging("ingest_data")


def ingest_latest_raw_data() -> dict:
    settings = get_settings()
    init_db()

    customers = clean_customers(pd.read_csv(settings.raw_data_dir / "customers_latest.csv"))
    activities = clean_activity_events(pd.read_csv(settings.raw_data_dir / "activity_events_latest.csv"))
    transactions = clean_transaction_events(pd.read_csv(settings.raw_data_dir / "transaction_events_latest.csv"))
    tickets = clean_support_tickets(pd.read_csv(settings.raw_data_dir / "support_tickets_latest.csv"))
    customers["created_at"] = pd.Timestamp.utcnow()

    quality = [
        add_data_quality_flags(customers, "customers"),
        add_data_quality_flags(activities, "activity_events"),
        add_data_quality_flags(transactions, "transaction_events"),
        add_data_quality_flags(tickets, "support_tickets"),
    ]

    engine = get_engine()
    with engine.begin() as connection:
        for table in [
            "activity_events",
            "transaction_events",
            "support_tickets",
            "customers",
        ]:
            connection.execute(text(f"DELETE FROM {table}"))
        connection.execute(text("DROP TABLE IF EXISTS customer_features"))

    customers.to_sql("customers", engine, if_exists="append", index=False)
    activities.to_sql("activity_events", engine, if_exists="append", index=False)
    transactions.to_sql("transaction_events", engine, if_exists="append", index=False)
    tickets.to_sql("support_tickets", engine, if_exists="append", index=False)

    features = build_feature_table(customers, activities, transactions, tickets)
    features.to_sql("customer_features", engine, if_exists="replace", index=False)

    path, version = save_dataframe_version(features, "customer_features")
    session = SessionLocal()
    session.add(
        DatasetVersion(
            dataset_name="customer_features",
            version=version,
            file_path=str(path),
            row_count=len(features),
            metadata_json={"quality": quality},
        )
    )
    session.commit()
    session.close()

    logger.info("Ingested raw data and created feature store version %s", version)
    return {"version": version, "path": str(path), "quality": quality, "row_count": len(features)}


if __name__ == "__main__":
    ingest_latest_raw_data()
