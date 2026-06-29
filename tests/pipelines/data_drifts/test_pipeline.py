"""Tests for the 'data_drifts' pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit_mlops.pipelines.data_drifts.nodes import (
    _psi_single,
    compute_psi_metrics,
    raise_drift_alerts,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def drift_params() -> dict:
    return {
        "psi": {
            "n_bins": 10,
            "bucket_type": "quantiles",
            "small_drift_threshold": 0.10,
            "major_drift_threshold": 0.25,
        },
        "alerts": {"max_drifted_share_critical": 0.30},
    }


@pytest.fixture
def reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(1000),
            "feature_a": rng.normal(loc=0, scale=1, size=1000),
            "feature_b": rng.uniform(0, 100, size=1000),
        }
    )


# --------------------------------------------------------------------------- #
# PSI computation                                                              #
# --------------------------------------------------------------------------- #
class TestPsiSingle:
    def test_psi_is_zero_when_distributions_are_identical(self, reference_df):
        arr = reference_df["feature_a"].to_numpy()
        psi = _psi_single(arr, arr.copy(), n_bins=10, bucket_type="quantiles")
        # Two identical samples → counts identical → PSI must be 0 exactly
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_psi_grows_when_distribution_shifts(self):
        rng = np.random.default_rng(seed=0)
        reference = rng.normal(loc=0, scale=1, size=5000)
        # Strong location shift
        current = rng.normal(loc=3, scale=1, size=5000)
        psi = _psi_single(reference, current, n_bins=10, bucket_type="quantiles")
        # A 3σ shift should comfortably exceed the "major" PSI threshold (0.25)
        assert psi > 0.25

    def test_psi_handles_constant_reference_without_crashing(self):
        constant = np.zeros(100)
        psi = _psi_single(constant, constant, n_bins=10, bucket_type="quantiles")
        # Degenerate but defined; either 0 or NaN — never an exception
        assert psi == 0.0 or np.isnan(psi)


# --------------------------------------------------------------------------- #
# compute_psi_metrics — node-level contract                                    #
# --------------------------------------------------------------------------- #
class TestComputePsiMetrics:
    def test_returns_one_row_per_feature_excluding_id(self, reference_df, drift_params):
        out = compute_psi_metrics(reference_df, reference_df.copy(), drift_params)
        # SK_ID_CURR is the row identifier and must be excluded
        assert "SK_ID_CURR" not in out["feature"].values
        assert set(out["feature"]) == {"feature_a", "feature_b"}

    def test_identical_data_yields_no_drift(self, reference_df, drift_params):
        out = compute_psi_metrics(reference_df, reference_df.copy(), drift_params)
        assert (out["drift_level"] == "none").all()
        assert (out["psi"].abs() < 1e-6).all()

    def test_strong_shift_is_flagged_major(self, drift_params):
        rng = np.random.default_rng(seed=0)
        ref = pd.DataFrame({"x": rng.normal(0, 1, size=5000)})
        cur = pd.DataFrame({"x": rng.normal(3, 1, size=5000)})
        out = compute_psi_metrics(ref, cur, drift_params)
        assert out.loc[0, "drift_level"] == "major"


# --------------------------------------------------------------------------- #
# raise_drift_alerts — status decision                                         #
# --------------------------------------------------------------------------- #
class TestRaiseDriftAlerts:
    def test_status_ok_when_no_features_drifted(self, drift_params):
        psi_df = pd.DataFrame(
            [
                {"feature": "a", "psi": 0.01, "drift_level": "none"},
                {"feature": "b", "psi": 0.02, "drift_level": "none"},
            ]
        )
        alerts = raise_drift_alerts(psi_df, {}, drift_params)
        assert alerts["status"] == "ok"
        assert alerts["drifted_features_count"] == 0

    def test_status_warning_for_small_drift(self, drift_params):
        # 1 small drift out of 10 features → 10% drift share, below the 30%
        # critical threshold, so the run is warning (not critical).
        psi_df = pd.DataFrame(
            [{"feature": "a", "psi": 0.15, "drift_level": "small"}]
            + [{"feature": f"g{i}", "psi": 0.01, "drift_level": "none"} for i in range(9)]
        )
        alerts = raise_drift_alerts(psi_df, {}, drift_params)
        assert alerts["status"] == "warning"
        assert alerts["drifted_features_count"] == 1

    def test_status_critical_when_any_major_feature(self, drift_params):
        psi_df = pd.DataFrame(
            [
                {"feature": "a", "psi": 0.40, "drift_level": "major"},
                {"feature": "b", "psi": 0.02, "drift_level": "none"},
            ]
        )
        alerts = raise_drift_alerts(psi_df, {}, drift_params)
        assert alerts["status"] == "critical"
        assert alerts["major_drift_features_count"] == 1

    def test_status_critical_when_too_many_small_drifts(self, drift_params):
        # 40% of features have small drift → above the 30% critical share
        psi_df = pd.DataFrame(
            [{"feature": f"f{i}", "psi": 0.15, "drift_level": "small"} for i in range(4)]
            + [{"feature": f"g{i}", "psi": 0.01, "drift_level": "none"} for i in range(6)]
        )
        alerts = raise_drift_alerts(psi_df, {}, drift_params)
        assert alerts["status"] == "critical"
        assert alerts["drifted_share"] == pytest.approx(0.4)

    def test_alert_payload_is_json_serialisable(self, drift_params):
        import json

        psi_df = pd.DataFrame([{"feature": "a", "psi": 0.40, "drift_level": "major"}])
        alerts = raise_drift_alerts(psi_df, {}, drift_params)
        # Round-trip through JSON must succeed without TypeError
        json.dumps(alerts)
