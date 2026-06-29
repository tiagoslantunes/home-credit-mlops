"""FastAPI serving layer for the Home Credit Default Risk model.

Two scoring endpoints:

  * ``POST /predict``      — accepts a JSON payload with the engineered
                              feature vector (51 columns, matching the model's
                              ``feature_names_in_``). This is the contract a
                              real feature service / batch pipeline would use.
  * ``POST /predict/raw``  — accepts a raw application_train.csv-style payload
                              and applies the training cleaning + feature
                              engineering steps internally. Convenience endpoint
                              for demos and callers without a feature service.

Plus utility endpoints:
  * ``GET /health``        — liveness probe + summary of loaded artefacts.
  * ``GET /model/schema``  — the ordered list of feature names ``/predict``
                              expects, so a client can introspect at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

from app.preprocessing import raw_to_features
from app.schemas import (
    FeaturesPayload,
    HealthResponse,
    PredictionResponse,
    RawApplicationPayload,
)
from home_credit_mlops.pipelines.model_predict.nodes import score_features

logger = logging.getLogger(__name__)

# Resolve paths from an env var so the same image works in dev and prod.
PROJECT_ROOT = Path(os.getenv("HOME_CREDIT_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "production_model.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "data" / "06_models" / "decision_threshold.json"
CLEANING_PARAMS_PATH = PROJECT_ROOT / "data" / "04_feature" / "cleaning_params.pkl"
CONF_DIR = PROJECT_ROOT / "conf" / "base"


class _State:
    """Container for artefacts loaded at startup."""

    model = None
    threshold: float | None = None
    columns: list[str] | None = None
    cleaning_params: Dict[str, Any] | None = None
    cleaning_node_params: Dict[str, Any] | None = None
    feat_node_params: Dict[str, Any] | None = None


state = _State()


def _load_artefacts() -> None:
    """Load model, threshold, cleaning artefact, and node parameters from disk."""
    with open(MODEL_PATH, "rb") as fh:
        state.model = pickle.load(fh)
    state.columns = list(state.model.feature_names_in_)

    with open(THRESHOLD_PATH) as fh:
        threshold_payload = json.load(fh)
    state.threshold = float(
        threshold_payload["threshold"] if isinstance(threshold_payload, dict) else threshold_payload
    )

    with open(CLEANING_PARAMS_PATH, "rb") as fh:
        state.cleaning_params = pickle.load(fh)

    with open(CONF_DIR / "parameters_data_cleaning.yml") as fh:
        state.cleaning_node_params = yaml.safe_load(fh)["data_cleaning"]
    with open(CONF_DIR / "parameters_data_feat_engineering.yml") as fh:
        state.feat_node_params = yaml.safe_load(fh)["data_feat_engineering"]

    logger.info(
        "Serving artefacts loaded: model=%s, threshold=%.4f, n_features=%d",
        type(state.model).__name__,
        state.threshold,
        len(state.columns),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_artefacts()
    yield


app = FastAPI(
    title="Home Credit Default Risk API",
    description="Serving layer for the calibrated LightGBM credit-default model.",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Health + introspection                                                       #
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.model is not None else "loading",
        model_loaded=state.model is not None,
        n_features=len(state.columns or []),
        threshold=state.threshold,
    )


@app.get("/model/schema", tags=["meta"])
def model_schema() -> Dict[str, Any]:
    """The exact ordered feature names ``POST /predict`` expects."""
    if state.columns is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")
    return {"feature_names": state.columns, "n_features": len(state.columns)}


# --------------------------------------------------------------------------- #
# Scoring endpoints                                                            #
# --------------------------------------------------------------------------- #
@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict_features(payload: FeaturesPayload) -> PredictionResponse:
    """Score an already-engineered feature vector."""
    if state.model is None or state.threshold is None or state.columns is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    features = pd.DataFrame([payload.features])
    try:
        result = score_features(features, state.model, decision_threshold=state.threshold)
    except ValueError as exc:
        # Missing or malformed columns — caller error, not a server bug.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    row = result.iloc[0]
    return PredictionResponse(
        probability=float(row["probability"]),
        predicted_class=int(row["predicted_class"]),
        threshold=state.threshold,
    )


@app.post("/predict/raw", response_model=PredictionResponse, tags=["scoring"])
def predict_raw(payload: RawApplicationPayload) -> PredictionResponse:
    """Score a raw application payload: pipeline runs cleaning + feature engineering internally."""
    if state.model is None or state.threshold is None or state.columns is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    try:
        features, fallback = raw_to_features(
            raw=payload.application,
            cleaning_params=state.cleaning_params,
            cleaning_node_params=state.cleaning_node_params,
            feat_node_params=state.feat_node_params,
            model_columns=state.columns,
        )
        result = score_features(features, state.model, decision_threshold=state.threshold)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    row = result.iloc[0]
    return PredictionResponse(
        probability=float(row["probability"]),
        predicted_class=int(row["predicted_class"]),
        threshold=state.threshold,
        fallback_features=fallback,
    )
