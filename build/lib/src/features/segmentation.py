from __future__ import annotations

import pandas as pd


def assign_user_segments(features: pd.DataFrame) -> pd.Series:
    sessions_threshold = features["sessions_last_30d"].quantile(0.75)
    frequency_threshold = features["frequency_per_week"].quantile(0.75)

    dormant = (features["recency_days"] > 21) | (features["sessions_last_30d"] <= 1)
    power = (
        (features["sessions_last_30d"] >= sessions_threshold)
        & (features["frequency_per_week"] >= frequency_threshold)
        & (features["feature_adoption_ratio"] >= 0.6)
    )

    segment = pd.Series("active_user", index=features.index)
    segment.loc[dormant] = "dormant_user"
    segment.loc[power] = "power_user"
    return segment
