"""
Integration tests for the data_feat_engineering Kedro pipeline.

Verifies that create_pipeline() returns a correctly wired Pipeline:
correct node names, expected inputs/outputs, and that the DAG executes
end-to-end on synthetic data using Kedro's MemoryDataset runner.
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from home_credit_mlops.pipelines.data_feat_engineering.pipeline import create_pipeline

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_app_df(n: int, seed: int, has_target: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "SK_ID_CURR":             rng.integers(100_000, 200_000, n),
            "AMT_INCOME_TOTAL":       rng.uniform(20_000, 200_000, n),
            "AMT_CREDIT":             rng.uniform(50_000, 500_000, n),
            "AMT_ANNUITY":            rng.uniform(5_000, 50_000, n),
            "AMT_GOODS_PRICE":        rng.uniform(30_000, 400_000, n),
            "DAYS_BIRTH":             rng.integers(-25_000, -7_000, n),
            "DAYS_EMPLOYED":          rng.integers(-5_000, 0, n),
            "DAYS_REGISTRATION":      rng.integers(-10_000, 0, n),
            "DAYS_ID_PUBLISH":        rng.integers(-5_000, 0, n),
            "DAYS_LAST_PHONE_CHANGE": rng.integers(-3_000, 0, n),
            "CNT_CHILDREN":           rng.integers(0, 4, n),
            "CNT_FAM_MEMBERS":        rng.integers(1, 6, n),
            "EXT_SOURCE_2":           rng.uniform(0, 1, n),
            "EXT_SOURCE_3":           rng.uniform(0, 1, n),
            "CODE_GENDER":            rng.choice(["M", "F"], n),
            "OCCUPATION_TYPE":        rng.choice(
                ["Laborers", "Sales staff", "Core staff", "Managers",
                 "Drivers", "High skill tech staff", "Accountants",
                 "Medicine staff", "Security staff", "Cooking staff", "Cleaning staff"],
                n,
            ),
            "FLAG_OWN_CAR":           rng.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY":        rng.choice(["Y", "N"], n),
            "NAME_CONTRACT_TYPE":     rng.choice(["Cash loans", "Revolving loans"], n),
            "NAME_EDUCATION_TYPE":    rng.choice(
                ["Higher education", "Secondary / secondary special",
                 "Incomplete higher", "Lower secondary"],
                n,
            ),
            "FLAG_DOCUMENT_3":        rng.integers(0, 2, n),
            "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 5, n),
        }
    )
    if has_target:
        df.insert(1, "TARGET", rng.integers(0, 2, n))
    return df


PARAMS = {
    "target_column": "TARGET",
    "id_column": "SK_ID_CURR",
    "ext_source_cols": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "binary_map": {
        "FLAG_OWN_CAR":    {"N": 0, "Y": 1},
        "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
    },
    "ohe_cardinality_threshold": 10,
    "target_encoding_cols": ["OCCUPATION_TYPE"],
    "target_encoding_smoothing": 10,
    "target_encoding_n_folds": 5,
    "target_encoding_random_state": 42,
    "feature_selection_method": "rfe",
    "n_features_to_select": 5,
    "rfe_step": 2,
    "rfe_estimator_params": {
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
        "n_jobs": 1,
    },
    "hopsworks_project": "test_project",
    "hopsworks_feature_group_name": "home_credit_features",
    "hopsworks_feature_group_version": 1,
    "hopsworks_feature_view_name": "home_credit_feature_view",
    "hopsworks_feature_view_version": 1,
    "hopsworks_host": "eu-west.cloud.hopsworks.ai",
}


@pytest.fixture()
def catalog():
    return DataCatalog(
        {
            "application_train_cleaned":      MemoryDataset(_make_app_df(100, 42, True)),
            "application_validation_cleaned": MemoryDataset(_make_app_df(25, 77, True)),
            "application_test_cleaned":       MemoryDataset(_make_app_df(20, 88, False)),
            "params:data_feat_engineering":   MemoryDataset(PARAMS),
            # Outputs — pre-registered so MemoryDataset accepts writes
            "application_train_features":     MemoryDataset(),
            "application_validation_features": MemoryDataset(),
            "application_test_features":      MemoryDataset(),
            "X_train_data":                   MemoryDataset(),
            "y_train_data":                   MemoryDataset(),
            "X_val_data":                     MemoryDataset(),
            "y_val_data":                     MemoryDataset(),
            "X_test_data":                    MemoryDataset(),
            "best_columns":                   MemoryDataset(),
            "feature_store_metadata":         MemoryDataset(),
        }
    )


# ── Pipeline structure tests ──────────────────────────────────────────────────

class TestPipelineStructure:

    def test_pipeline_has_three_nodes(self):
        pipe = create_pipeline()
        assert len(pipe.nodes) == 3

    def test_node_names(self):
        pipe = create_pipeline()
        names = {n.name for n in pipe.nodes}
        assert names == {"create_features_node", "feature_selection_node", "feature_store_node"}

    def test_create_features_node_outputs(self):
        pipe = create_pipeline()
        node = next(n for n in pipe.nodes if n.name == "create_features_node")
        assert set(node.outputs) == {
            "application_train_features",
            "application_validation_features",
            "application_test_features",
        }

    def test_select_features_node_outputs(self):
        pipe = create_pipeline()
        node = next(n for n in pipe.nodes if n.name == "feature_selection_node")
        assert set(node.outputs) == {
            "X_train_data", "y_train_data",
            "X_val_data",   "y_val_data",
            "X_test_data",  "best_columns",
        }

    def test_feature_store_node_output(self):
        pipe = create_pipeline()
        node = next(n for n in pipe.nodes if n.name == "feature_store_node")
        assert "feature_store_metadata" in node.outputs


# ── End-to-end run tests ──────────────────────────────────────────────────────

class TestPipelineRun:

    def test_pipeline_runs_without_error(self, catalog):
        """create_features + select_features nodes run; feature_store_node skipped (no API key)."""
        pipe = create_pipeline().only_nodes(
            "create_features_node",
            "feature_selection_node",
        )
        SequentialRunner().run(pipe, catalog)

    def test_x_train_has_id_and_selected_features(self, catalog):
        pipe = create_pipeline().only_nodes(
            "create_features_node",
            "feature_selection_node",
        )
        SequentialRunner().run(pipe, catalog)
        X_train = catalog.load("X_train_data")
        assert "SK_ID_CURR" in X_train.columns
        assert X_train.shape[1] == PARAMS["n_features_to_select"] + 1

    def test_feature_store_node_skips_gracefully(self):
        """Without HOPSWORKS_API_KEY to_feature_store returns a 'skipped' metadata dict."""
        from home_credit_mlops.pipelines.data_feat_engineering.nodes import to_feature_store

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "feat_a": rng.uniform(0, 1, 3)})
        y = pd.DataFrame({"TARGET": [0, 1, 0]})

        # Patch load_dotenv so the real .env key is not reloaded, and clear HOPSWORKS_API_KEY
        env_without_key = {k: v for k, v in os.environ.items() if k != "HOPSWORKS_API_KEY"}
        with patch("dotenv.load_dotenv"), patch.dict("os.environ", env_without_key, clear=True):
            metadata = to_feature_store(X, y, X, y, X, PARAMS)
        assert metadata["status"] == "skipped"

    def test_best_columns_persisted(self, catalog):
        pipe = create_pipeline().only_nodes(
            "create_features_node",
            "feature_selection_node",
        )
        SequentialRunner().run(pipe, catalog)
        best_cols = catalog.load("best_columns")
        assert isinstance(best_cols, list)
        assert len(best_cols) == PARAMS["n_features_to_select"]

    def test_y_train_shape(self, catalog):
        pipe = create_pipeline().only_nodes(
            "create_features_node",
            "feature_selection_node",
        )
        SequentialRunner().run(pipe, catalog)
        y_train = catalog.load("y_train_data")
        assert y_train.shape == (100, 1)
        assert "TARGET" in y_train.columns


# ── Hopsworks upload tests (mocked) ──────────────────────────────────────────

def _hopsworks_fixtures():
    """Return (mock_fg, mock_fs, mock_project) wired together."""
    from unittest.mock import MagicMock
    mock_fg      = MagicMock()
    mock_fs      = MagicMock()
    mock_project = MagicMock()
    mock_fs.get_or_create_feature_group.return_value = mock_fg
    mock_project.get_feature_store.return_value = mock_fs
    return mock_fg, mock_fs, mock_project


def _xy(n: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "SK_ID_CURR": range(n),
        "feat_a": rng.uniform(0, 1, n),
        "feat_b": rng.uniform(0, 1, n),
    })
    y = pd.DataFrame({"TARGET": rng.integers(0, 2, n)})
    return X, y


class TestToFeatureStoreHopsworks:

    def _run(self, extra_params=None):
        from home_credit_mlops.pipelines.data_feat_engineering.nodes import to_feature_store
        mock_fg, mock_fs, mock_project = _hopsworks_fixtures()
        X, y = _xy()
        params = {**PARAMS, **(extra_params or {})}
        with patch("hopsworks.login", return_value=mock_project), \
             patch("dotenv.load_dotenv"), \
             patch.dict("os.environ", {"HOPSWORKS_API_KEY": "fake-key"}):
            metadata = to_feature_store(X, y, X, y, X, params)
        return metadata, mock_fg, mock_fs

    def test_upload_returns_success(self):
        metadata, _, _ = self._run()
        assert metadata["status"] == "success"

    def test_feature_group_insert_called(self):
        _, mock_fg, _ = self._run()
        assert mock_fg.insert.called

    def test_upload_df_has_split_column(self):
        _, mock_fg, _ = self._run()
        inserted_df = mock_fg.insert.call_args_list[0][0][0]
        assert "split" in inserted_df.columns
        assert set(inserted_df["split"].unique()) == {"train", "validation", "test"}

    def test_upload_df_has_event_time(self):
        _, mock_fg, _ = self._run()
        inserted_df = mock_fg.insert.call_args_list[0][0][0]
        assert "event_time" in inserted_df.columns

    def test_metadata_contains_row_counts(self):
        metadata, _, _ = self._run()
        assert metadata["n_rows_train"] == 15
        assert metadata["n_rows_val"]   == 15
        assert metadata["n_rows_test"]  == 15

    def test_last_batch_waits_for_job(self):
        _, mock_fg, _ = self._run()
        last_call = mock_fg.insert.call_args_list[-1]
        assert last_call[1]["write_options"]["wait_for_job"] is True

    def test_feature_view_created_when_enabled(self):
        _, _, mock_fs = self._run({"enable_feature_view": True})
        assert mock_fs.get_or_create_feature_view.called
        call_kwargs = mock_fs.get_or_create_feature_view.call_args[1]
        assert call_kwargs["name"] == PARAMS["hopsworks_feature_view_name"]

    def test_feature_view_skipped_when_disabled(self):
        _, _, mock_fs = self._run({"enable_feature_view": False})
        assert not mock_fs.get_or_create_feature_view.called

    def test_feature_view_in_metadata_when_enabled(self):
        metadata, _, _ = self._run({"enable_feature_view": True})
        assert "feature_view" in metadata
        assert metadata["feature_view"] == PARAMS["hopsworks_feature_view_name"]
