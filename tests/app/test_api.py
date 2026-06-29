"""Smoke + contract tests for the FastAPI serving layer.

Uses FastAPI's ``TestClient`` so the tests run in-process: no need for a live
``uvicorn`` server and no port conflicts in CI.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client() -> TestClient:
    # The ``with`` form ensures FastAPI's lifespan (model loading) runs before
    # the first request and is torn down after the last.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def features_row() -> dict:
    """First row of X_val.csv — already-engineered features the model trained on."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "05_model_input" / "X_val.csv", nrows=1)
    return df.iloc[0].to_dict()


@pytest.fixture(scope="module")
def raw_application_row() -> dict:
    """First row of application_train.csv as a raw payload, NaN→None for JSON."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "01_raw" / "application_train.csv", nrows=1)
    raw = df.iloc[0].to_dict()
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Meta endpoints                                                               #
# --------------------------------------------------------------------------- #
class TestMeta:
    def test_health_reports_loaded_model(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["n_features"] == 51
        assert 0.0 < body["threshold"] < 1.0

    def test_model_schema_lists_51_features(self, client):
        resp = client.get("/model/schema")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_features"] == 51
        assert isinstance(body["feature_names"], list)
        assert body["feature_names"][0] == "SK_ID_CURR"


# --------------------------------------------------------------------------- #
# /predict — engineered features contract                                      #
# --------------------------------------------------------------------------- #
class TestPredictFeatures:
    def test_returns_valid_probability_and_class(self, client, features_row):
        resp = client.post("/predict", json={"features": features_row})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["predicted_class"] in (0, 1)

    def test_response_matches_batch_pipeline_first_row(self, client, features_row):
        # The batch pipeline wrote predictions.csv earlier — the API must agree
        # with it bit-for-bit when given the same row.
        resp = client.post("/predict", json={"features": features_row})
        api_prob = resp.json()["probability"]
        batch = pd.read_csv(PROJECT_ROOT / "data" / "07_model_output" / "predictions.csv", nrows=1)
        batch_prob = float(batch.loc[0, "probability"])
        assert api_prob == pytest.approx(batch_prob, rel=1e-6)

    def test_rejects_payload_missing_required_columns(self, client):
        resp = client.post("/predict", json={"features": {"SK_ID_CURR": 1}})
        assert resp.status_code == 422

    def test_rejects_malformed_payload_shape(self, client):
        # Forgetting the "features" wrapper → 422 from Pydantic.
        resp = client.post("/predict", json={"SK_ID_CURR": 1})
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /predict/raw — raw application contract                                      #
# --------------------------------------------------------------------------- #
class TestPredictRaw:
    def test_returns_valid_probability_with_te_fallback(self, client, raw_application_row):
        resp = client.post("/predict/raw", json={"application": raw_application_row})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert 0.0 <= body["probability"] <= 1.0
        # OCCUPATION_TYPE_TE and ORGANIZATION_TYPE_TE always use the fallback because
        # the trained encoder maps are not persisted by the upstream pipeline.
        assert "OCCUPATION_TYPE_TE" in body["fallback_features"]
        assert "ORGANIZATION_TYPE_TE" in body["fallback_features"]

    def test_rejects_payload_missing_application_wrapper(self, client):
        resp = client.post("/predict/raw", json={"AMT_INCOME_TOTAL": 1000})
        assert resp.status_code == 422
