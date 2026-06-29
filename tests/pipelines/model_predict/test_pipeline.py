"""Tests for the 'model_predict' pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit_mlops.pipelines.model_predict.nodes import (
    evaluate_predictions,
    get_production_columns,
    predict,
    score_features,
)


# --------------------------------------------------------------------------- #
# Tiny stub model — mimics the sklearn-style API we rely on                    #
# --------------------------------------------------------------------------- #
class _StubModel:
    """Predicts P(y=1) ≈ x['x1'] (clipped to [0, 1])."""

    def __init__(self, feature_names):
        # sklearn stores feature names as an ndarray on the fitted estimator
        self.feature_names_in_ = np.array(feature_names)

    def predict_proba(self, X):
        p = X.iloc[:, 0].to_numpy().clip(0, 1)
        return np.column_stack([1 - p, p])


@pytest.fixture
def stub_model() -> _StubModel:
    return _StubModel(["x1", "x2"])


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100, 101, 102, 103],
            "x1": [0.1, 0.6, 0.9, 0.4],
            "x2": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def predict_params() -> dict:
    return {"id_column": "SK_ID_CURR", "log_metrics_when_labels_available": True}


# --------------------------------------------------------------------------- #
# get_production_columns                                                       #
# --------------------------------------------------------------------------- #
class TestGetProductionColumns:
    def test_reads_columns_from_fitted_model(self, stub_model):
        assert get_production_columns(stub_model) == ["x1", "x2"]

    def test_raises_when_model_not_fit_on_dataframe(self):
        class _NotFit:
            pass

        with pytest.raises(AttributeError, match="feature_names_in_"):
            get_production_columns(_NotFit())


# --------------------------------------------------------------------------- #
# score_features                                                               #
# --------------------------------------------------------------------------- #
class TestScoreFeatures:
    def test_output_shape_and_columns(self, stub_model, features):
        out = score_features(features, stub_model, decision_threshold=0.5, id_column="SK_ID_CURR")
        assert list(out.columns) == ["SK_ID_CURR", "probability", "predicted_class"]
        assert len(out) == len(features)

    def test_probabilities_are_in_unit_interval(self, stub_model, features):
        out = score_features(features, stub_model, decision_threshold=0.5)
        assert (out["probability"] >= 0).all()
        assert (out["probability"] <= 1).all()

    def test_class_is_zero_or_one(self, stub_model, features):
        out = score_features(features, stub_model, decision_threshold=0.5)
        assert set(out["predicted_class"].unique()).issubset({0, 1})

    def test_threshold_affects_decision(self, stub_model, features):
        low_t = score_features(features, stub_model, decision_threshold=0.05)
        high_t = score_features(features, stub_model, decision_threshold=0.95)
        # A more permissive threshold can only ever predict the same or MORE positives
        assert low_t["predicted_class"].sum() >= high_t["predicted_class"].sum()

    def test_id_column_is_preserved_in_output(self, stub_model, features):
        out = score_features(features, stub_model, decision_threshold=0.5, id_column="SK_ID_CURR")
        pd.testing.assert_series_equal(
            out["SK_ID_CURR"].reset_index(drop=True),
            features["SK_ID_CURR"].reset_index(drop=True),
            check_names=False,
        )

    def test_raises_on_missing_required_columns(self, stub_model):
        bad = pd.DataFrame({"x1": [0.5]})  # x2 missing
        with pytest.raises(ValueError, match="missing"):
            score_features(bad, stub_model, decision_threshold=0.5)

    def test_sanitises_whitespace_in_column_names(self, stub_model):
        # LightGBM replaces spaces with underscores in feature_names_in_;
        # raw CSVs may still hold the un-sanitised originals.
        model = _StubModel(["x1", "Cash_loans"])
        raw = pd.DataFrame({"x1": [0.5], "Cash loans": [1]})
        # Must NOT raise — sanitisation should bridge the gap
        out = score_features(raw, model, decision_threshold=0.5)
        assert len(out) == 1

    def test_reorders_columns_to_model_order(self, stub_model):
        # Pass columns in the wrong order; expect the model still sees them as fit-time order.
        features_unordered = pd.DataFrame({"x2": [10.0], "x1": [0.7]})
        out = score_features(features_unordered, stub_model, decision_threshold=0.5)
        # Stub predicts proba ≈ x1; reordered model would have returned ≈10 (clipped to 1.0).
        assert out.loc[0, "probability"] == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# predict node (wrapper)                                                       #
# --------------------------------------------------------------------------- #
class TestPredictNode:
    def test_predict_accepts_threshold_as_dict(self, stub_model, features, predict_params):
        out = predict(features, stub_model, {"threshold": 0.5}, predict_params)
        assert "probability" in out.columns

    def test_predict_accepts_threshold_as_float(self, stub_model, features, predict_params):
        out = predict(features, stub_model, 0.5, predict_params)
        assert "probability" in out.columns


# --------------------------------------------------------------------------- #
# evaluate_predictions                                                         #
# --------------------------------------------------------------------------- #
class TestEvaluatePredictions:
    def test_returns_expected_metric_keys(self):
        predictions = pd.DataFrame(
            {
                "probability": [0.1, 0.7, 0.9, 0.4],
                "predicted_class": [0, 1, 1, 0],
            }
        )
        y_true = pd.DataFrame({"y": [0, 1, 1, 0]})
        metrics = evaluate_predictions(predictions, y_true, decision_threshold=0.5)

        expected_keys = {
            "roc_auc",
            "pr_auc",
            "accuracy_at_threshold",
            "precision_at_threshold",
            "recall_at_threshold",
            "f1_at_threshold",
            "threshold",
            "positive_rate_predicted",
            "positive_rate_actual",
            "n_rows",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_perfect_predictions_give_roc_auc_one(self):
        predictions = pd.DataFrame(
            {
                "probability": [0.05, 0.95, 0.95, 0.05],
                "predicted_class": [0, 1, 1, 0],
            }
        )
        y_true = pd.DataFrame({"y": [0, 1, 1, 0]})
        metrics = evaluate_predictions(predictions, y_true, decision_threshold=0.5)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["accuracy_at_threshold"] == pytest.approx(1.0)
