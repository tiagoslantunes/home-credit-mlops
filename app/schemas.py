"""Pydantic schemas for the serving API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FeaturesPayload(BaseModel):
    """Already-engineered feature vector — same schema as a row of X_val.csv.

    In production this is what an upstream feature service (Feast, Tecton, etc.)
    would deliver to the model. The /predict endpoint expects this contract so
    the model API stays small and independent of the feature pipeline.
    """

    model_config = ConfigDict(extra="allow")  # allow forward-compatible additions

    features: Dict[str, float] = Field(
        ...,
        description=(
            "Mapping from feature name to value. Must include every feature listed "
            "by ``GET /model/schema``."
        ),
        examples=[{"SK_ID_CURR": 100002, "AMT_INCOME_TOTAL": 202500.0}],
    )


class RawApplicationPayload(BaseModel):
    """Raw application_train.csv-style payload — pre-engineering.

    The /predict/raw endpoint accepts this, applies the same cleaning +
    feature-engineering steps the training pipeline does, and scores. Convenient
    for demos and for callers that don't have a feature service in front of the
    model.
    """

    model_config = ConfigDict(extra="allow")

    application: Dict[str, Optional[float | str | int]] = Field(
        ...,
        description="Raw application columns; missing values may be omitted.",
        examples=[
            {
                "SK_ID_CURR": 100002,
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Single / not married",
                "OCCUPATION_TYPE": "Laborers",
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "EXT_SOURCE_2": 0.262949,
                "EXT_SOURCE_3": 0.139376,
            }
        ],
    )


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="P(default = 1) from the calibrated model.", ge=0.0, le=1.0)
    predicted_class: int = Field(..., description="1 if probability ≥ threshold else 0.", ge=0, le=1)
    threshold: float = Field(..., description="Decision threshold used (chosen on the validation set).")
    fallback_features: List[str] = Field(
        default_factory=list,
        description=(
            "Feature names where a fallback default was used because the value "
            "could not be derived from the raw payload (e.g. target-encoded "
            "categories not seen at training time). Empty when no fallback was "
            "needed."
        ),
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    threshold: float | None
