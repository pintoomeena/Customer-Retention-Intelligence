from __future__ import annotations

import pandas as pd

from src.models.serving import score_frame
from src.utils.logging import configure_logging


logger = configure_logging("model_scoring")


def score_customers(
    features: pd.DataFrame,
    persist: bool = True,
    batch_id: str | None = None,
    model_source: str | None = None,
) -> pd.DataFrame:
    scored, _actions, metadata = score_frame(
        features,
        model_source=model_source,
        persist=persist,
        batch_id=batch_id,
    )
    resolved_batch_id = batch_id or (str(scored["batch_id"].iloc[0]) if not scored.empty else "n/a")
    logger.info(
        "Scored %s customers for batch %s using source %s",
        len(scored),
        resolved_batch_id,
        metadata["source_key"],
    )
    return scored
