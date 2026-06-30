"""
Unit tests for the data_split node.

split_data performs a stratified train/validation split BEFORE cleaning. Doing
the split first is what guarantees that every imputation statistic (learned later
in data_cleaning) is fit on the train partition only — the validation rows must
never influence any cleaning decision. These tests pin down the split's
contract: correct sizes, preserved schema, preserved class balance,
reproducibility, and no row leaking into both partitions.

The sample is synthetic (built with numpy), so the tests are fast and do not
depend on any file under data/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit_mlops.pipelines.data_split.nodes import split_data
from home_credit_mlops.pipelines.data_split.pipeline import create_pipeline

# Mirrors conf/base/parameters_data_split.yml. Defined inline so the unit test
# does not depend on Kedro's config loader.
SPLIT_PARAMS = {
    "target_column": "TARGET",
    "test_size": 0.2,       # 20% goes to validation
    "random_state": 42,     # fixed seed -> deterministic split
    "stratify": True,       # keep the positive rate equal across partitions
}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    # 100 rows with ~8% positives, mimicking the (heavily imbalanced) target.
    # The exact 92/8 layout lets the stratification assertions use a tight bound.
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        "SK_ID_CURR": range(1, n + 1),
        "TARGET": ([0] * 92) + ([1] * 8),
        "FEATURE_A": np.random.normal(0, 1, n),
        "FEATURE_B": np.random.uniform(0, 1, n),
    })


class TestSplitData:
    def test_sizes_add_up(self, sample_df):
        # No rows are dropped or duplicated: train + validation == original.
        train, val = split_data(sample_df, SPLIT_PARAMS)
        assert len(train) + len(val) == len(sample_df)

    def test_validation_fraction_is_approximate(self, sample_df):
        # Validation should be ~test_size of the total (±2 rows for rounding).
        _, val = split_data(sample_df, SPLIT_PARAMS)
        expected = len(sample_df) * SPLIT_PARAMS["test_size"]
        assert abs(len(val) - expected) <= 2

    def test_columns_preserved(self, sample_df):
        # Splitting partitions rows, never columns: both frames keep every column.
        train, val = split_data(sample_df, SPLIT_PARAMS)
        assert set(train.columns) == set(sample_df.columns)
        assert set(val.columns) == set(sample_df.columns)

    def test_target_present_in_both_splits(self, sample_df):
        # The label must travel with both partitions (train to fit, val to evaluate).
        train, val = split_data(sample_df, SPLIT_PARAMS)
        assert "TARGET" in train.columns
        assert "TARGET" in val.columns

    def test_stratification_preserves_class_ratio(self, sample_df):
        # The whole point of stratify=True: both partitions keep the original
        # positive rate, so a rare-event problem doesn't starve one split of 1s.
        train, val = split_data(sample_df, SPLIT_PARAMS)
        orig_rate = sample_df["TARGET"].mean()
        assert abs(train["TARGET"].mean() - orig_rate) < 0.05
        assert abs(val["TARGET"].mean() - orig_rate) < 0.05

    def test_reproducibility(self, sample_df):
        # Same random_state -> byte-identical splits, so runs are reproducible.
        train1, val1 = split_data(sample_df, SPLIT_PARAMS)
        train2, val2 = split_data(sample_df, SPLIT_PARAMS)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)

    def test_no_overlap_between_train_and_validation(self, sample_df):
        # No applicant may be in both partitions, otherwise validation would be
        # contaminated with rows the model already saw — a form of leakage.
        train, val = split_data(sample_df, SPLIT_PARAMS)
        train_ids = set(train["SK_ID_CURR"])
        val_ids = set(val["SK_ID_CURR"])
        assert train_ids.isdisjoint(val_ids)

    def test_index_is_reset(self, sample_df):
        # split_data resets the index so downstream CSV writes don't carry a stale
        # original index; both frames should be a clean 0..n-1 RangeIndex.
        train, val = split_data(sample_df, SPLIT_PARAMS)
        assert list(train.index) == list(range(len(train)))
        assert list(val.index) == list(range(len(val)))

    def test_without_stratification(self, sample_df):
        # The stratify flag is optional: with stratify=False the split must still
        # run and keep all rows (just without the class-balance guarantee).
        params = {**SPLIT_PARAMS, "stratify": False}
        train, val = split_data(sample_df, params)
        assert len(train) + len(val) == len(sample_df)


class TestPipelineWiring:
    """Covers create_pipeline (the split_data node is tested directly above)."""

    def test_create_pipeline_wires_split_node(self):
        # The pipeline must expose the split node and produce both partitions.
        pipe = create_pipeline()
        names = {node.name for node in pipe.nodes}
        assert "split_application_train_node" in names
        outputs = set().union(*(node.outputs for node in pipe.nodes))
        assert {"application_train_split", "application_validation_split"} <= outputs
