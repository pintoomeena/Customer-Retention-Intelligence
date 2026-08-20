from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class CustomerRequest(BaseModel):
    customer_id: str | None = Field(default=None, description="Customer identifier from the feature store.")
    model_source: str = Field(default="auto", description="Model source to use: auto, demo, or kaggle_cell2cell.")
    features: dict[str, Any] | None = Field(
        default=None,
        description="Optional direct feature payload for ad-hoc scoring.",
    )
    persist: bool = Field(default=False, description="Persist prediction and action records.")

    @model_validator(mode="after")
    def validate_payload(self) -> "CustomerRequest":
        if not self.customer_id and not self.features:
            raise ValueError("Provide either customer_id or features.")
        return self


class ActivateModelRequest(BaseModel):
    model_source: str = Field(description="Model source to activate.")
