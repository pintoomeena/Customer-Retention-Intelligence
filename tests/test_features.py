from __future__ import annotations

import pandas as pd

from src.features.engineering import build_feature_table


def test_build_feature_table_creates_core_features():
    customers = pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "signup_date": "2025-01-01",
                "churned_at": None,
                "plan": "Basic",
                "country": "US",
                "monthly_revenue": 29.0,
                "status": "active",
                "acquisition_channel": "organic",
            }
        ]
    )
    activities = pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "event_time": "2026-01-10",
                "session_id": "S1",
                "feature_name": "core_dashboard",
                "duration_minutes": 15.0,
                "events_in_session": 3,
            }
        ]
    )
    transactions = pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "transaction_time": "2026-01-05",
                "amount": 29.0,
                "status": "succeeded",
                "days_late": 0,
            }
        ]
    )
    tickets = pd.DataFrame(
        [
            {
                "ticket_id": "T1",
                "customer_id": "C1",
                "opened_at": "2025-12-20",
                "resolved_at": "2025-12-21",
                "priority": "low",
                "category": "general",
                "satisfaction_score": 4.5,
            }
        ]
    )

    features = build_feature_table(
        customers,
        activities,
        transactions,
        tickets,
        reference_date=pd.Timestamp("2026-01-15"),
    )

    assert len(features) == 1
    assert {"recency_days", "feature_adoption_ratio", "engagement_score", "target_churn"}.issubset(
        set(features.columns)
    )
    assert features.loc[0, "customer_segment"] in {"active_user", "dormant_user", "power_user"}
