from __future__ import annotations

import pandas as pd

from src.retention.action_engine import recommend_action, risk_segment


def test_risk_segment_thresholds():
    assert risk_segment(0.2) == "low_risk"
    assert risk_segment(0.5) == "medium_risk"
    assert risk_segment(0.9) == "high_risk"


def test_recommend_action_returns_complete_payload():
    customer = pd.Series(
        {
            "customer_id": "C1",
            "payment_risk_flag": 1,
            "support_risk_flag": 0,
            "feature_adoption_ratio": 0.2,
            "activity_decay_flag": 1,
            "customer_segment": "dormant_user",
        }
    )
    action = recommend_action(customer, churn_probability=0.82, explanation={"risk_factors": []}, batch_id="b1")

    assert action["risk_segment"] == "high_risk"
    assert action["action_type"] in {"discount_offer", "holdout_control"}
    assert "payload_json" in action
