from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.utils.config import get_settings


def risk_segment(probability: float) -> str:
    settings = get_settings()
    if probability >= settings.risk_high_threshold:
        return "high_risk"
    if probability >= settings.risk_medium_threshold:
        return "medium_risk"
    return "low_risk"


def assign_experiment_group(customer_id: str, batch_id: str) -> str:
    bucket = int(hashlib.md5(f"{customer_id}:{batch_id}".encode("utf-8")).hexdigest(), 16) % 100
    return "control" if bucket < 50 else "treatment"


def recommend_action(
    customer: pd.Series | dict[str, Any],
    churn_probability: float,
    explanation: dict[str, Any] | None = None,
    batch_id: str = "manual",
) -> dict[str, Any]:
    row = customer if isinstance(customer, pd.Series) else pd.Series(customer)
    segment = risk_segment(churn_probability)
    experiment_group = assign_experiment_group(str(row["customer_id"]), batch_id)
    reasons = explanation.get("risk_factors", []) if explanation else []
    reason_summary = "; ".join(item["message"] for item in reasons[:2]) or "General churn risk threshold exceeded."

    if segment == "high_risk":
        if row.get("payment_risk_flag", 0) == 1:
            action = ("discount_offer", "email", {"offer_percent": 15})
        elif row.get("support_risk_flag", 0) == 1:
            action = ("support_outreach", "phone", {"owner": "customer_success"})
        elif row.get("feature_adoption_ratio", 0.0) < 0.35:
            action = ("onboarding_reactivation", "email", {"journey": "feature_adoption"})
        else:
            action = ("executive_checkin", "phone", {"owner": "retention_team"})
    elif segment == "medium_risk":
        if row.get("activity_decay_flag", 0) == 1:
            action = ("reminder_email", "email", {"template": "usage_drop_detected"})
        elif row.get("feature_adoption_ratio", 0.0) < 0.5:
            action = ("feature_nudge", "in_app", {"feature_pack": "high_value_features"})
        else:
            action = ("value_recap", "email", {"template": "roi_summary"})
    else:
        if row.get("customer_segment") == "power_user":
            action = ("upsell_opportunity", "email", {"offer": "premium_addon"})
        else:
            action = ("loyalty_nurture", "email", {"template": "customer_story"})

    action_type, channel, payload = action
    status = "queued"

    if segment in {"high_risk", "medium_risk"} and experiment_group == "control":
        action_type = "holdout_control"
        channel = "none"
        payload = {"note": "A/B control group - no intervention"}
        status = "suppressed"

    return {
        "customer_id": str(row["customer_id"]),
        "churn_probability": round(float(churn_probability), 4),
        "risk_segment": segment,
        "action_type": action_type,
        "channel": channel,
        "payload_json": payload,
        "rationale": reason_summary,
        "experiment_group": experiment_group,
        "status": status,
        "batch_id": batch_id,
    }
