from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DATE_COLUMNS = {
    "customers": ["signup_date", "churned_at"],
    "activity_events": ["event_time"],
    "transaction_events": ["transaction_time"],
    "support_tickets": ["opened_at", "resolved_at"],
}


def _normalize_dates(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def cap_iqr_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    if series.empty:
        return series
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return series.clip(lower=lower, upper=upper)


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_dates(df.copy(), DATE_COLUMNS["customers"])
    df = df.drop_duplicates(subset=["customer_id"]).copy()
    df["plan"] = df["plan"].fillna("Basic").str.title()
    df["country"] = df["country"].fillna("Unknown").str.upper()
    df["acquisition_channel"] = df["acquisition_channel"].fillna("unknown").str.lower()
    df["status"] = df["status"].fillna("active").str.lower()
    df["monthly_revenue"] = cap_iqr_outliers(df["monthly_revenue"].fillna(df["monthly_revenue"].median()))
    return df


def clean_activity_events(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_dates(df.copy(), DATE_COLUMNS["activity_events"])
    df = df.drop_duplicates(subset=["customer_id", "session_id", "feature_name", "event_time"]).copy()
    df["feature_name"] = df["feature_name"].fillna("core_dashboard").str.lower()
    df["duration_minutes"] = cap_iqr_outliers(df["duration_minutes"].fillna(df["duration_minutes"].median()))
    df["events_in_session"] = (
        df["events_in_session"].fillna(1).astype(int).clip(lower=1, upper=50)
    )
    return df.dropna(subset=["customer_id", "event_time", "session_id"])


def clean_transaction_events(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_dates(df.copy(), DATE_COLUMNS["transaction_events"])
    df = df.drop_duplicates(subset=["customer_id", "transaction_time", "amount", "status"]).copy()
    df["status"] = df["status"].fillna("succeeded").str.lower()
    df["days_late"] = df["days_late"].fillna(0).astype(int).clip(lower=0, upper=90)
    df["amount"] = cap_iqr_outliers(df["amount"].fillna(df["amount"].median()))
    return df.dropna(subset=["customer_id", "transaction_time"])


def clean_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_dates(df.copy(), DATE_COLUMNS["support_tickets"])
    df = df.drop_duplicates(subset=["ticket_id"]).copy()
    df["priority"] = df["priority"].fillna("medium").str.lower()
    df["category"] = df["category"].fillna("general").str.lower()
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())
    df["satisfaction_score"] = df["satisfaction_score"].fillna(3.0).clip(lower=1.0, upper=5.0)
    return df.dropna(subset=["ticket_id", "customer_id", "opened_at"])


def add_data_quality_flags(df: pd.DataFrame, name: str) -> dict[str, float]:
    missing_rate = float(df.isna().mean().mean()) if not df.empty else 0.0
    duplicate_rate = float(df.duplicated().mean()) if not df.empty else 0.0
    return {
        "table": name,
        "row_count": int(len(df)),
        "missing_rate": round(missing_rate, 4),
        "duplicate_rate": round(duplicate_rate, 4),
    }
