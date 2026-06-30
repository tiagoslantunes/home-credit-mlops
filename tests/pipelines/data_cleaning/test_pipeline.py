"""
Unit tests for the data_cleaning nodes.

Cleaning is a fit/apply pair (same philosophy as the week_04 example, where the
encoder is fit on train and reused on the batch):

  * ``fit_cleaning(train, params, numerical_rules)`` runs on the TRAIN split only.
    It learns the columns to drop (high-missing / highly-correlated /
    near-constant) and the imputation statistics (medians, modes), returning them
    as a single ``cleaning_params`` dict (the "artifact").
  * ``apply_cleaning(df, artifact, params)`` is pure replay: it transforms any
    frame (train, validation or a future serving batch) using only the artifact,
    recomputing nothing.

What these tests collectively guarantee:
  - no nulls remain after cleaning (incl. an extreme production batch),
  - the structural drops fire (missing / correlation / near-constant),
  - the DAYS_EMPLOYED sentinel is turned into NaN + a flag,
  - imputation statistics come from TRAIN only (no leakage into validation),
  - value capping and integer dtype casting are applied.

The sample is fully synthetic (built with numpy below), so the tests are fast
and do not depend on any file under data/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit_mlops.pipelines.data_cleaning.nodes import (
    _high_correlation_columns,
    _near_constant_columns,
    apply_cleaning,
    fit_cleaning,
)
from home_credit_mlops.pipelines.data_cleaning.pipeline import create_pipeline

# Mirrors conf/base/parameters_data_cleaning.yml. Kept here as a constant so the
# tests are self-contained and do not load the real YAML (a unit test should not
# depend on Kedro's config loader).
CLEANING_PARAMS = {
    "id_columns": ["SK_ID_CURR"],          # never dropped, never imputed
    "target_column": "TARGET",             # never dropped, never imputed
    "missing_threshold": 0.45,             # drop columns with > 45% missing
    "corr_threshold": 0.9,                 # drop the 2nd column of each |r| > 0.9 pair
    "near_constant_threshold": 0.95,       # drop columns whose dominant value >= 95%
    "days_employed_col": "DAYS_EMPLOYED",
    "days_employed_flag": "DAYS_EMPLOYED_ANOM",
    "days_employed_sentinel": 365243,      # "not employed" placeholder -> NaN + flag
    # filled with 0 (absence of a bureau/social-circle record == zero events)
    "zero_fill_cols": [
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
    ],
    # imputed with the train median (listed mainly for documentation/drop-protection)
    "median_fill_cols": ["AMT_ANNUITY", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_EMPLOYED"],
    # integer-count columns -> train median rounded to the nearest integer
    "median_int_cols": ["CNT_FAM_MEMBERS", "DAYS_LAST_PHONE_CHANGE"],
    # categorical column where missingness is itself a signal -> explicit category
    "constant_fill": {"OCCUPATION_TYPE": "Unknown"},
    # conceptually-integer columns coerced back to int64 at the end (no NaN left)
    "cast_int_cols": [
        "DAYS_EMPLOYED",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ],
    "apply_value_bounds": True,            # enable the defensive clip to the rule bounds
}

# Subset of params:numerical_rules (defined in parameters_data_quality.yml) that
# is relevant to the columns present in the sample. These are domain rules, not
# statistics derived from the data, so the cap never depends on the test data.
NUMERICAL_RULES = {
    "EXT_SOURCE_2": {"min_value": 0, "max_value": 1},
    "EXT_SOURCE_3": {"min_value": 0, "max_value": 1},
    "AMT_ANNUITY": {"min_value": 1},
    "CNT_FAM_MEMBERS": {"min_value": 1},
    "AMT_REQ_CREDIT_BUREAU_MON": {"min_value": 0},
    "OBS_30_CNT_SOCIAL_CIRCLE": {"min_value": 0},
    "DAYS_LAST_PHONE_CHANGE": {"max_value": 0},
}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic 200-row sample crafted so that every cleaning behaviour fires.

    Each block of columns is engineered to trigger one specific code path:
      * DAYS_EMPLOYED        -> 20% set to the 365243 sentinel (tests the flag + NaN swap)
      * AMT_ANNUITY/EXT_*    -> have NaN, tests median imputation
      * NAME_TYPE_SUITE      -> has NaN, tests the generic categorical-mode fallback
      * OCCUPATION_TYPE      -> has NaN, tests the explicit "Unknown" fill
      * AMT_REQ_* / *_CIRCLE -> have NaN, tests the zero-fill
      * CNT_FAM_MEMBERS, ... -> have NaN, tests the integer-median fill
      * CORR_A / CORR_B      -> r ~ 0.99, tests the correlation drop (CORR_B goes)
      * NEAR_CONST           -> 96% one value, tests the near-constant drop
      * JUNK_HIGH_MISSING    -> ~50% missing, tests the high-missing drop
    """
    rng = np.random.default_rng(0)  # fixed seed -> deterministic, reproducible tests
    n = 200

    # 20% of the rows carry the "not employed" sentinel instead of a real value.
    days_employed = rng.integers(-15000, -100, n).astype(float)
    days_employed[: int(0.2 * n)] = 365243

    # CORR_B is CORR_A plus tiny noise -> correlation ~0.99 (one of them is redundant).
    corr_a = rng.normal(0, 1, n)
    corr_b = corr_a + rng.normal(0, 0.01, n)

    # 96% zeros, 4% ones -> dominant value above the 95% near-constant threshold.
    near_const = np.zeros(n)
    near_const[: int(0.04 * n)] = 1

    df = pd.DataFrame({
        "SK_ID_CURR": range(1, n + 1),
        "TARGET": ([0] * 184) + ([1] * 16),  # ~8% positives, mimics the class imbalance
        "DAYS_EMPLOYED": days_employed,
        "AMT_ANNUITY": rng.uniform(5_000, 50_000, n),
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "EXT_SOURCE_3": rng.uniform(0, 1, n),
        "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Family", "Spouse, partner"], n, p=[0.8, 0.15, 0.05]),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Core staff", "Managers"], n),
        "AMT_REQ_CREDIT_BUREAU_MON": rng.integers(0, 4, n).astype(float),
        "AMT_REQ_CREDIT_BUREAU_QRT": rng.integers(0, 4, n).astype(float),
        "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 6, n).astype(float),
        "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 4, n).astype(float),
        "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, n).astype(float),
        "DEF_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, n).astype(float),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "DAYS_LAST_PHONE_CHANGE": rng.integers(-3000, 0, n).astype(float),
        "CORR_A": corr_a,
        "CORR_B": corr_b,
        "NEAR_CONST": near_const,
        "JUNK_HIGH_MISSING": rng.normal(0, 1, n),
    })

    # Punch holes (NaN) into specific columns: (column, how many rows to blank).
    for col, k in [
        ("AMT_ANNUITY", 10), ("EXT_SOURCE_2", 15), ("EXT_SOURCE_3", 30),
        ("NAME_TYPE_SUITE", 5), ("OCCUPATION_TYPE", 60),
        ("AMT_REQ_CREDIT_BUREAU_MON", 25), ("AMT_REQ_CREDIT_BUREAU_QRT", 25),
        ("AMT_REQ_CREDIT_BUREAU_YEAR", 25), ("OBS_30_CNT_SOCIAL_CIRCLE", 5),
        ("DEF_30_CNT_SOCIAL_CIRCLE", 5), ("DEF_60_CNT_SOCIAL_CIRCLE", 5),
        ("CNT_FAM_MEMBERS", 2), ("DAYS_LAST_PHONE_CHANGE", 1),
    ]:
        df.loc[rng.choice(n, k, replace=False), col] = np.nan
    # ~50% missing -> above the 45% threshold, so this column must be dropped.
    df.loc[rng.choice(n, n // 2, replace=False), "JUNK_HIGH_MISSING"] = np.nan

    return df


@pytest.fixture
def artifact(raw_df) -> dict:
    """The learned cleaning artifact (fit on the sample as if it were the train split).
    Shared by every test that needs to apply or inspect the fitted statistics."""
    return fit_cleaning(raw_df, CLEANING_PARAMS, NUMERICAL_RULES)


class TestFitCleaning:
    """Covers the 'fit' half: what fit_cleaning learns from the train split."""

    def test_returns_dict_with_expected_keys(self, artifact):
        # The artifact is what gets pickled and replayed; it must expose the four
        # pieces apply_cleaning relies on, otherwise replay would silently break.
        assert isinstance(artifact, dict)
        for key in ("cols_to_drop", "numeric_medians", "categorical_modes", "value_bounds"):
            assert key in artifact

    def test_drops_high_missing(self, artifact):
        # JUNK_HIGH_MISSING has ~50% NaN (> the 45% threshold) -> must be dropped.
        assert "JUNK_HIGH_MISSING" in artifact["cols_to_drop"]

    def test_drops_high_correlation(self, artifact):
        # CORR_A/CORR_B are ~0.99 correlated; the rule keeps the first, drops the
        # second column of the pair. So CORR_B goes and CORR_A stays.
        assert "CORR_B" in artifact["cols_to_drop"]
        assert "CORR_A" not in artifact["cols_to_drop"]

    def test_drops_near_constant(self, artifact):
        # NEAR_CONST is 96% a single value -> above the 95% near-constant threshold.
        assert "NEAR_CONST" in artifact["cols_to_drop"]

    def test_id_and_target_never_dropped(self, artifact):
        # The id and target are protected; losing them would break joins/training.
        assert "SK_ID_CURR" not in artifact["cols_to_drop"]
        assert "TARGET" not in artifact["cols_to_drop"]

    def test_bounds_come_from_rules(self, artifact):
        # The cap bounds must be copied verbatim from numerical_rules (domain rules),
        # not derived from the data — this is what keeps the cap leakage-free.
        assert artifact["value_bounds"]["EXT_SOURCE_2"] == {"min": 0, "max": 1}


class TestApplyCleaning:
    """Covers the 'apply' half: replaying the artifact on a frame."""

    def test_no_nulls(self, raw_df, artifact):
        # The headline guarantee: a cleaned frame is fully imputed, no NaN anywhere.
        out = apply_cleaning(raw_df, artifact, CLEANING_PARAMS)
        assert [c for c in out.columns if out[c].isnull().any()] == []

    def test_dropped_columns_absent(self, raw_df, artifact):
        # The columns flagged in fit must actually be gone from the output frame.
        out = apply_cleaning(raw_df, artifact, CLEANING_PARAMS)
        for col in ("JUNK_HIGH_MISSING", "CORR_B", "NEAR_CONST"):
            assert col not in out.columns

    def test_days_employed_sentinel(self, raw_df, artifact):
        # After cleaning: no 365243 left, the anomaly flag exists, and the flag's
        # 1-count equals how many sentinels were in the raw input.
        out = apply_cleaning(raw_df, artifact, CLEANING_PARAMS)
        assert (out["DAYS_EMPLOYED"] == 365243).sum() == 0
        assert "DAYS_EMPLOYED_ANOM" in out.columns
        assert out["DAYS_EMPLOYED_ANOM"].sum() == int((raw_df["DAYS_EMPLOYED"] == 365243).sum())

    def test_zero_fill(self, raw_df, artifact):
        # Rows that were NaN in a zero-fill column must become exactly 0.
        out = apply_cleaning(raw_df, artifact, CLEANING_PARAMS)
        idx = raw_df["AMT_REQ_CREDIT_BUREAU_MON"].isna()
        assert (out.loc[idx, "AMT_REQ_CREDIT_BUREAU_MON"] == 0).all()

    def test_constant_fill_occupation(self, raw_df, artifact):
        # OCCUPATION_TYPE uses an explicit category ("Unknown"), not the mode,
        # because the missingness itself is a predictive signal.
        out = apply_cleaning(raw_df, artifact, CLEANING_PARAMS)
        idx = raw_df["OCCUPATION_TYPE"].isna()
        assert (out.loc[idx, "OCCUPATION_TYPE"] == "Unknown").all()

    def test_integer_dtype_cast(self, raw_df, artifact):
        # Count/day columns that pandas promoted to float (because of NaN) must be
        # restored to int64 once imputation removed every NaN.
        out = apply_cleaning(raw_df, artifact, CLEANING_PARAMS)
        for col in ("DAYS_EMPLOYED", "AMT_REQ_CREDIT_BUREAU_MON", "OBS_30_CNT_SOCIAL_CIRCLE"):
            assert out[col].dtype == "int64"

    def test_capping_to_bounds(self, raw_df, artifact):
        # Inject an out-of-range value (1.5 > max 1) and confirm it is clipped.
        # This is the defensive guard that protects serving against drifted inputs.
        batch = raw_df.copy()
        batch.loc[batch.index[:5], "EXT_SOURCE_2"] = 1.5
        out = apply_cleaning(batch, artifact, CLEANING_PARAMS)
        assert out["EXT_SOURCE_2"].max() <= 1.0
        assert out["EXT_SOURCE_2"].min() >= 0.0

    def test_same_columns_in_train_and_validation(self, raw_df, artifact):
        # Replaying on a smaller 'validation' frame must yield the identical column
        # set/order as train, so downstream feature engineering sees one schema.
        train_out = apply_cleaning(raw_df, artifact, CLEANING_PARAMS)
        val_out = apply_cleaning(raw_df.head(40), artifact, CLEANING_PARAMS)
        assert list(train_out.columns) == list(val_out.columns)


class TestNoLeakage:
    """The key methodological guarantee: validation is imputed with TRAIN stats."""

    def test_imputation_uses_train_median(self, artifact):
        # Build a validation frame whose own EXT_SOURCE_3 median (~0.96) is clearly
        # different from the train median. The NaN row must be filled with the TRAIN
        # median (stored in the artifact), proving the validation data never feeds
        # the imputation — i.e. no information leaks from validation into the fit.
        train_median = artifact["numeric_medians"]["EXT_SOURCE_3"]
        val = pd.DataFrame({
            "SK_ID_CURR": [9001, 9002, 9003, 9004],
            "EXT_SOURCE_3": [np.nan, 0.95, 0.96, 0.97],  # own median ~0.96
        })
        out = apply_cleaning(val, artifact, CLEANING_PARAMS)
        assert out["EXT_SOURCE_3"].iloc[0] == pytest.approx(train_median)
        assert train_median != pytest.approx(0.96)  # sanity: train != validation median


class TestProductionFallback:
    """The generic fallback that makes the pipeline safe for arbitrary batches."""

    def test_batch_has_no_nulls_even_with_everything_missing(self, raw_df, artifact):
        # Worst case: a serving batch with NaN scattered across EVERY feature column.
        # The generic numeric-median / categorical-mode fallback must still return a
        # frame with zero NaN, so scoring never crashes on missing inputs.
        batch = raw_df.copy()
        cols = [c for c in batch.columns if c not in ("SK_ID_CURR", "TARGET")]
        for col in cols:
            batch.loc[batch.sample(frac=0.5, random_state=1).index, col] = np.nan
        out = apply_cleaning(batch, artifact, CLEANING_PARAMS)
        assert int(out.isnull().sum().sum()) == 0


class TestEdgeCases:
    """Defensive branches that the normal sample doesn't reach."""

    def test_correlation_with_fewer_than_two_numeric_columns(self):
        # With < 2 numeric columns there is no pair to correlate, so the detector
        # must short-circuit and return an empty drop list (instead of erroring).
        df = pd.DataFrame({"ONLY_NUMERIC": [1.0, 2.0, 3.0], "CAT": ["a", "b", "c"]})
        assert _high_correlation_columns(df, threshold=0.9, protected=set()) == []

    def test_near_constant_skips_all_nan_column(self):
        # An all-NaN column has no observed values, so it cannot be "near-constant";
        # the detector must skip it rather than crash on an empty value_counts.
        df = pd.DataFrame({"ALL_NAN": [np.nan, np.nan, np.nan], "CONST": [1, 1, 1]})
        result = _near_constant_columns(df, threshold=0.95, protected=set())
        assert "ALL_NAN" not in result
        assert "CONST" in result  # the genuinely constant column is still flagged

    def test_warns_when_nulls_remain(self, raw_df, artifact, caplog):
        # A column unseen at fit time has no learned median/mode, so it cannot be
        # imputed and stays NaN. apply_cleaning must log a warning in that case.
        import logging

        batch = raw_df.copy()
        batch["UNSEEN_COLUMN"] = np.nan  # never seen by fit_cleaning
        with caplog.at_level(logging.WARNING):
            out = apply_cleaning(batch, artifact, CLEANING_PARAMS)
        assert out["UNSEEN_COLUMN"].isnull().all()
        assert any("missing values remain" in r.message for r in caplog.records)


class TestPipelineWiring:
    """Covers create_pipeline (the node functions are tested directly elsewhere)."""

    def test_create_pipeline_has_fit_and_apply_nodes(self):
        # The pipeline must wire one fit node plus an apply node for train,
        # validation and the test/holdout batch.
        pipe = create_pipeline()
        names = {node.name for node in pipe.nodes}
        assert {
            "fit_cleaning_node",
            "clean_train_node",
            "clean_validation_node",
            "clean_test_node",
        } <= names
