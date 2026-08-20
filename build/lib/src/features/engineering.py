from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.features.segmentation import assign_user_segments


ALL_FEATURES = [
    "core_dashboard",
    "automation",
    "collaboration",
    "reporting",
    "billing",
    "analytics",
]


def _weekly_activity_slope(group: pd.DataFrame, reference_date: pd.Timestamp) -> float:
    if group.empty:
        return 0.0
    weekly = (
        group.assign(week=((reference_date - group["event_time"]).dt.days // 7).astype(int))
        .query("week >= 0 and week < 8")
        .groupby("week")["session_id"]
        .nunique()
        .reindex(range(8), fill_value=0)
        .sort_index(ascending=False)
    )
    x = np.arange(len(weekly))
    slope = float(np.polyfit(x, weekly.values, 1)[0]) if weekly.nunique() > 1 else 0.0
    return slope


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    return numerator / denominator.replace(0, np.nan) if isinstance(denominator, pd.Series) else numerator / denominator


def build_feature_table(
    customers: pd.DataFrame,
    activities: pd.DataFrame,
    transactions: pd.DataFrame,
    tickets: pd.DataFrame,
    reference_date: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    reference_date = pd.Timestamp(reference_date or pd.Timestamp.utcnow().normalize())
    customers = customers.copy()
    activities = activities.copy()
    transactions = transactions.copy()
    tickets = tickets.copy()

    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    customers["churned_at"] = pd.to_datetime(customers["churned_at"])
    activities["event_time"] = pd.to_datetime(activities["event_time"])
    transactions["transaction_time"] = pd.to_datetime(transactions["transaction_time"])
    tickets["opened_at"] = pd.to_datetime(tickets["opened_at"])
    tickets["resolved_at"] = pd.to_datetime(tickets["resolved_at"])

    feature_rows = customers[[
        "customer_id",
        "signup_date",
        "plan",
        "country",
        "monthly_revenue",
        "status",
        "acquisition_channel",
    ]].copy()
    feature_rows["snapshot_date"] = reference_date
    feature_rows["cohort_month"] = feature_rows["signup_date"].dt.to_period("M").astype(str)
    feature_rows["tenure_days"] = (reference_date - feature_rows["signup_date"]).dt.days.clip(lower=1)

    last_30d = activities[activities["event_time"] >= reference_date - pd.Timedelta(days=30)]
    last_84d = activities[activities["event_time"] >= reference_date - pd.Timedelta(days=84)]
    last_90d = activities[activities["event_time"] >= reference_date - pd.Timedelta(days=90)]

    activity_summary = activities.groupby("customer_id").agg(
        last_activity_at=("event_time", "max"),
        total_sessions=("session_id", "nunique"),
        total_minutes=("duration_minutes", "sum"),
    )
    activity_summary["recency_days"] = (
        reference_date - activity_summary["last_activity_at"]
    ).dt.days.clip(lower=0)

    sessions_30d = last_30d.groupby("customer_id")["session_id"].nunique().rename("sessions_last_30d")
    active_days_30d = last_30d.assign(event_date=last_30d["event_time"].dt.date).groupby("customer_id")[
        "event_date"
    ].nunique().rename("active_days_last_30d")
    sessions_84d = last_84d.groupby("customer_id")["session_id"].nunique().rename("sessions_last_84d")
    frequency_per_week = (sessions_84d / 12.0).rename("frequency_per_week")
    duration_30d = last_30d.groupby("customer_id")["duration_minutes"].sum().rename("duration_minutes_last_30d")
    unique_features_90d = last_90d.groupby("customer_id")["feature_name"].nunique().rename("feature_usage_count_90d")
    feature_adoption_ratio = (unique_features_90d / len(ALL_FEATURES)).rename("feature_adoption_ratio")
    drop_off_points = (len(ALL_FEATURES) - unique_features_90d).clip(lower=0).rename("drop_off_points")
    events_per_session = (
        last_90d.groupby("customer_id")["events_in_session"].mean().rename("avg_events_per_session")
    )

    slopes = (
        activities.groupby("customer_id")
        .apply(lambda group: _weekly_activity_slope(group, reference_date), include_groups=False)
        .rename("activity_trend_slope")
    )

    activity_features = pd.concat(
        [
            activity_summary[["recency_days"]],
            sessions_30d,
            active_days_30d,
            frequency_per_week,
            duration_30d,
            unique_features_90d,
            feature_adoption_ratio,
            drop_off_points,
            events_per_session,
            slopes,
        ],
        axis=1,
    )

    tx_90d = transactions[transactions["transaction_time"] >= reference_date - pd.Timedelta(days=90)]
    tx_180d = transactions[transactions["transaction_time"] >= reference_date - pd.Timedelta(days=180)]

    financial_features = pd.concat(
        [
            tx_180d.groupby("customer_id")["days_late"].mean().rename("avg_payment_delay_days"),
            tx_180d.query("status != 'succeeded'").groupby("customer_id")["status"].count().rename(
                "failed_transactions_180d"
            ),
            tx_90d.query("status == 'succeeded'").groupby("customer_id")["amount"].sum().rename("monetary_value_90d"),
            tx_180d.groupby("customer_id")["amount"].mean().rename("avg_invoice_amount"),
        ],
        axis=1,
    )

    ticket_resolution_hours = (
        (tickets["resolved_at"] - tickets["opened_at"]).dt.total_seconds() / 3600
    ).where(tickets["resolved_at"].notna())
    tickets = tickets.assign(resolution_hours=ticket_resolution_hours)
    tickets_90d = tickets[tickets["opened_at"] >= reference_date - pd.Timedelta(days=90)]

    support_features = pd.concat(
        [
            tickets_90d.groupby("customer_id")["ticket_id"].count().rename("tickets_90d"),
            tickets_90d.groupby("customer_id")["resolution_hours"].mean().rename("avg_resolution_hours"),
            tickets_90d.groupby("customer_id")["satisfaction_score"].mean().rename("avg_ticket_csat"),
            tickets_90d[tickets_90d["resolved_at"].isna()].groupby("customer_id")["ticket_id"].count().rename(
                "open_tickets_90d"
            ),
        ],
        axis=1,
    )

    features = feature_rows.merge(activity_features, how="left", left_on="customer_id", right_index=True)
    features = features.merge(financial_features, how="left", left_on="customer_id", right_index=True)
    features = features.merge(support_features, how="left", left_on="customer_id", right_index=True)

    fill_map = {
        "recency_days": 999,
        "sessions_last_30d": 0,
        "active_days_last_30d": 0,
        "frequency_per_week": 0.0,
        "duration_minutes_last_30d": 0.0,
        "feature_usage_count_90d": 0,
        "feature_adoption_ratio": 0.0,
        "drop_off_points": len(ALL_FEATURES),
        "avg_events_per_session": 0.0,
        "activity_trend_slope": 0.0,
        "avg_payment_delay_days": 0.0,
        "failed_transactions_180d": 0,
        "monetary_value_90d": 0.0,
        "avg_invoice_amount": features["monthly_revenue"].median(),
        "tickets_90d": 0,
        "avg_resolution_hours": 0.0,
        "avg_ticket_csat": 4.0,
        "open_tickets_90d": 0,
    }
    features = features.fillna(fill_map)

    features["rfm_recency"] = features["recency_days"]
    features["rfm_frequency"] = features["sessions_last_30d"] + features["frequency_per_week"]
    features["rfm_monetary"] = features["monetary_value_90d"]
    features["engagement_score"] = (
        (features["sessions_last_30d"] * 0.35)
        + (features["active_days_last_30d"] * 0.2)
        + (features["feature_adoption_ratio"] * 30)
        + (features["avg_events_per_session"] * 0.5)
        + (features["activity_trend_slope"] * 2)
    )
    features["activity_decay_flag"] = (
        (features["activity_trend_slope"] < 0) & (features["recency_days"] > 14)
    ).astype(int)
    features["payment_risk_flag"] = (
        (features["avg_payment_delay_days"] > 5) | (features["failed_transactions_180d"] >= 2)
    ).astype(int)
    features["support_risk_flag"] = (
        (features["tickets_90d"] >= 2) & (features["avg_resolution_hours"] > 48)
    ).astype(int)
    features["churn_risk_heuristic"] = (
        features["recency_days"].clip(upper=60) / 60 * 0.35
        + (1 - features["feature_adoption_ratio"]).clip(lower=0) * 0.2
        + (features["failed_transactions_180d"].clip(upper=3) / 3) * 0.15
        + (features["avg_payment_delay_days"].clip(upper=20) / 20) * 0.1
        + (features["tickets_90d"].clip(upper=4) / 4) * 0.1
        + (features["activity_decay_flag"] * 0.1)
    ).round(4)

    features["customer_segment"] = assign_user_segments(features)
    features["target_churn"] = (features["status"] == "churned").astype(int)
    return features.sort_values("customer_id").reset_index(drop=True)


NUMERIC_FEATURES = [
    "tenure_days",
    "monthly_revenue",
    "recency_days",
    "sessions_last_30d",
    "active_days_last_30d",
    "frequency_per_week",
    "duration_minutes_last_30d",
    "feature_usage_count_90d",
    "feature_adoption_ratio",
    "drop_off_points",
    "avg_events_per_session",
    "activity_trend_slope",
    "avg_payment_delay_days",
    "failed_transactions_180d",
    "monetary_value_90d",
    "avg_invoice_amount",
    "tickets_90d",
    "avg_resolution_hours",
    "avg_ticket_csat",
    "open_tickets_90d",
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
    "engagement_score",
    "activity_decay_flag",
    "payment_risk_flag",
    "support_risk_flag",
    "churn_risk_heuristic",
]

CATEGORICAL_FEATURES = [
    "plan",
    "country",
    "acquisition_channel",
    "cohort_month",
    "customer_segment",
]
