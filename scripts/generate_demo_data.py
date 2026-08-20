from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.utils.config import get_settings
from src.utils.io import timestamp_slug
from src.utils.logging import configure_logging


logger = configure_logging("generate_demo_data")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_demo_data(n_customers: int = 1500, seed: int | None = None) -> dict[str, Path]:
    settings = get_settings()
    seed = settings.model_random_state if seed is None else seed
    rng = np.random.default_rng(seed)
    reference_date = pd.Timestamp.utcnow().normalize()
    raw_dir = settings.raw_data_dir
    version = timestamp_slug()

    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]
    plans = np.array(["Basic", "Pro", "Enterprise"])
    plan_probs = np.array([0.5, 0.35, 0.15])
    countries = np.array(["US", "IN", "GB", "DE", "AU", "CA"])
    acquisition_channels = np.array(["paid_search", "organic", "partner", "referral", "sales"])

    signup_offsets = rng.integers(30, 720, size=n_customers)
    signup_dates = reference_date - pd.to_timedelta(signup_offsets, unit="D")
    plan = rng.choice(plans, size=n_customers, p=plan_probs)
    country = rng.choice(countries, size=n_customers)
    acquisition_channel = rng.choice(acquisition_channels, size=n_customers)

    plan_revenue = {"Basic": 29.0, "Pro": 79.0, "Enterprise": 199.0}
    base_revenue = np.array([plan_revenue[p] for p in plan]) + rng.normal(0, 8, size=n_customers)

    engagement = rng.normal(0, 1, size=n_customers)
    adoption = rng.beta(2.2, 2.0, size=n_customers)
    payment_stress = rng.beta(1.7, 5.0, size=n_customers)
    support_burden = rng.beta(1.4, 3.0, size=n_customers)
    tenure_factor = np.clip(signup_offsets / 365.0, 0.1, 2.0)

    latent_risk = (
        -1.0
        - 0.8 * engagement
        - 1.2 * adoption
        + 2.0 * payment_stress
        + 1.5 * support_burden
        + 0.25 * (plan == "Basic").astype(int)
        - 0.2 * (plan == "Enterprise").astype(int)
        - 0.15 * tenure_factor
    )
    churn_probability = _sigmoid(latent_risk)
    churn_flag = rng.binomial(1, churn_probability).astype(bool)
    churned_days_ago = rng.integers(1, 30, size=n_customers)
    churned_at = pd.Series(
        np.where(churn_flag, (reference_date - pd.to_timedelta(churned_days_ago, unit="D")).astype(str), None)
    )
    status = np.where(churn_flag, "churned", "active")

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": signup_dates,
            "churned_at": pd.to_datetime(churned_at, errors="coerce"),
            "plan": plan,
            "country": country,
            "monthly_revenue": base_revenue.round(2),
            "status": status,
            "acquisition_channel": acquisition_channel,
        }
    )

    feature_names = np.array(["core_dashboard", "automation", "collaboration", "reporting", "billing", "analytics"])
    activity_rows: list[dict] = []
    for idx, customer_id in enumerate(customer_ids):
        weeks = 16
        plan_multiplier = {"Basic": 0.9, "Pro": 1.2, "Enterprise": 1.5}[plan[idx]]
        base_sessions = max(0.2, (2.8 + engagement[idx] * 1.1 + adoption[idx] * 2.0) * plan_multiplier)
        decay = 0.18 + churn_probability[idx] * 0.7 if churn_flag[idx] else 0.03 + payment_stress[idx] * 0.1
        for week in range(weeks):
            week_end = reference_date - pd.Timedelta(days=week * 7)
            expected = max(0.0, base_sessions - week * decay)
            sessions = rng.poisson(expected)
            for session_num in range(sessions):
                if churn_flag[idx] and week_end >= customers.loc[idx, "churned_at"]:
                    continue
                event_time = week_end - pd.Timedelta(days=int(rng.integers(0, 7)))
                duration = max(2.0, rng.normal(18 + engagement[idx] * 6, 5))
                adopted_feature_count = max(1, int(round(adoption[idx] * len(feature_names))))
                allowed_features = feature_names[:adopted_feature_count]
                activity_rows.append(
                    {
                        "customer_id": customer_id,
                        "event_time": event_time,
                        "session_id": f"{customer_id}-{week}-{session_num}",
                        "feature_name": str(rng.choice(allowed_features)),
                        "duration_minutes": round(duration, 2),
                        "events_in_session": int(rng.integers(1, 10)),
                    }
                )

    activities = pd.DataFrame(activity_rows)

    tx_rows: list[dict] = []
    for idx, customer_id in enumerate(customer_ids):
        for month in range(6):
            tx_date = reference_date - pd.DateOffset(days=30 * month)
            if churn_flag[idx] and pd.notna(customers.loc[idx, "churned_at"]) and tx_date > customers.loc[idx, "churned_at"]:
                continue
            failed = rng.random() < (payment_stress[idx] * 0.4)
            tx_rows.append(
                {
                    "customer_id": customer_id,
                    "transaction_time": tx_date,
                    "amount": round(max(10, base_revenue[idx] + rng.normal(0, 5)), 2),
                    "status": "failed" if failed else "succeeded",
                    "days_late": int(rng.integers(0, 15) if failed else rng.integers(0, 4)),
                }
            )
    transactions = pd.DataFrame(tx_rows)

    ticket_rows: list[dict] = []
    categories = np.array(["general", "billing", "bug", "training"])
    priorities = np.array(["low", "medium", "high"])
    for idx, customer_id in enumerate(customer_ids):
        ticket_count = rng.poisson(0.4 + support_burden[idx] * 2.6)
        for ticket_num in range(ticket_count):
            opened_at = reference_date - pd.Timedelta(days=int(rng.integers(1, 90)))
            unresolved = rng.random() < (support_burden[idx] * 0.2)
            resolution_hours = max(2, rng.normal(16 + support_burden[idx] * 48, 8))
            ticket_rows.append(
                {
                    "ticket_id": f"TICKET-{customer_id}-{ticket_num}",
                    "customer_id": customer_id,
                    "opened_at": opened_at,
                    "resolved_at": None if unresolved else opened_at + pd.Timedelta(hours=float(resolution_hours)),
                    "priority": str(rng.choice(priorities, p=[0.5, 0.35, 0.15])),
                    "category": str(rng.choice(categories)),
                    "satisfaction_score": round(float(np.clip(rng.normal(4 - support_burden[idx] * 1.8, 0.8), 1, 5)), 1),
                }
            )
    tickets = pd.DataFrame(ticket_rows)

    if not activities.empty:
        activities = pd.concat([activities, activities.sample(frac=0.01, random_state=seed)], ignore_index=True)
    if not transactions.empty:
        transactions.loc[transactions.sample(frac=0.005, random_state=seed).index, "amount"] *= 3

    outputs = {
        "customers": customers,
        "activity_events": activities,
        "transaction_events": transactions,
        "support_tickets": tickets,
    }

    paths: dict[str, Path] = {}
    for name, frame in outputs.items():
        versioned_path = raw_dir / f"{name}_{version}.csv"
        latest_path = raw_dir / f"{name}_latest.csv"
        frame.to_csv(versioned_path, index=False)
        frame.to_csv(latest_path, index=False)
        paths[name] = latest_path

    logger.info("Generated demo raw data for %s customers in %s", n_customers, raw_dir)
    return paths


if __name__ == "__main__":
    generate_demo_data()
