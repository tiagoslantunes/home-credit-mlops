"""
Tests for data_feat_engineering pipeline nodes.

Pattern: test node functions directly with small synthetic dataframes -
no real data files needed, no Kedro context required.
Run with: pytest tests/pipelines/data_feat_engineering/test_nodes.py
"""

import numpy as np
import pandas as pd
import pytest

from home_credit_mlops.pipelines.data_feat_engineering.nodes import (
    create_features,
    select_features,
)


# Fixtures


def _make_app_df(n: int = 80, seed: int = 42, has_target: bool = True) -> pd.DataFrame:
    """Minimal synthetic application dataframe. Includes NAME_* columns to exercise OHE."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "SK_ID_CURR":                rng.integers(100_000, 200_000, n),
            "AMT_INCOME_TOTAL":          rng.uniform(20_000, 200_000, n),
            "AMT_CREDIT":                rng.uniform(50_000, 500_000, n),
            "AMT_ANNUITY":               rng.uniform(5_000, 50_000, n),
            "AMT_GOODS_PRICE":           rng.uniform(30_000, 400_000, n),
            "DAYS_BIRTH":                rng.integers(-25_000, -7_000, n),
            "DAYS_EMPLOYED":             rng.integers(-5_000, 0, n),
            "DAYS_REGISTRATION":         rng.integers(-10_000, 0, n),
            "DAYS_ID_PUBLISH":           rng.integers(-5_000, 0, n),
            "DAYS_LAST_PHONE_CHANGE":    rng.integers(-3_000, 0, n),
            "CNT_CHILDREN":              rng.integers(0, 4, n),
            "CNT_FAM_MEMBERS":           rng.integers(1, 6, n),
            "EXT_SOURCE_1":              rng.uniform(0, 1, n),
            "EXT_SOURCE_2":              rng.uniform(0, 1, n),
            "EXT_SOURCE_3":              rng.uniform(0, 1, n),
            "CODE_GENDER":               rng.choice(["M", "F"], n),
            # >10 unique values -> survives OHE threshold, used for target encoding tests
            "OCCUPATION_TYPE":           rng.choice(["Laborers", "Sales staff", "Core staff",
                                                      "Managers", "Drivers", "High skill tech staff",
                                                      "Accountants", "Medicine staff", "Security staff",
                                                      "Cooking staff", "Cleaning staff"], n),
            "FLAG_OWN_CAR":              rng.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY":           rng.choice(["Y", "N"], n),
            # OHE-able columns (cardinality <= 10) - needed to exercise _encode_ohe path
            "NAME_CONTRACT_TYPE":        rng.choice(["Cash loans", "Revolving loans"], n),
            "NAME_EDUCATION_TYPE":       rng.choice(["Higher education", "Secondary / secondary special",
                                                      "Incomplete higher", "Lower secondary"], n),
            "FLAG_DOCUMENT_3":           rng.integers(0, 2, n),
            "FLAG_DOCUMENT_6":           rng.integers(0, 2, n),
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
        "FLAG_OWN_CAR":   {"N": 0, "Y": 1},
        "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
    },
    "ohe_cardinality_threshold": 10,
    "target_encoding_cols": [],
    "feature_selection_method": "rfe",
    "n_features_to_select": 5,
    "rfe_step": 2,
    "rfe_estimator_params": {
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
        "n_jobs": 1,
    },
}

# create_features tests


class TestCreateFeatures:

    def _run(self):
        return create_features(
            _make_app_df(80, seed=42, has_target=True),
            _make_app_df(20, seed=99, has_target=True),
            _make_app_df(15, seed=77, has_target=False),
            PARAMS,
        )

    def test_financial_ratios_are_created(self):
        train, val, test = self._run()
        for col in ("CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "PAYMENT_RATE",
                    "INCOME_PER_PERSON", "CHILDREN_RATIO"):
            assert col in train.columns, f"Missing: {col}"
            assert col in val.columns,   f"Missing in val: {col}"
            assert col in test.columns,  f"Missing in test: {col}"

    def test_age_years_is_positive(self):
        train, _, _ = self._run()
        assert "AGE_YEARS" in train.columns
        assert (train["AGE_YEARS"] > 0).all()

    def test_employed_years_non_negative(self):
        train, _, _ = self._run()
        assert "EMPLOYED_YEARS" in train.columns
        assert (train["EMPLOYED_YEARS"] >= 0).all()

    def test_phone_change_years_created(self):
        train, _, _ = self._run()
        assert "PHONE_CHANGE_YEARS" in train.columns
        assert (train["PHONE_CHANGE_YEARS"] >= 0).all()

    def test_ext_source_aggregates_created(self):
        train, _, _ = self._run()
        for col in ("EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX", "EXT_SOURCE_STD"):
            assert col in train.columns

    def test_document_count_created(self):
        train, _, _ = self._run()
        assert "DOCS_COUNT" in train.columns

    def test_bureau_inquiries_created(self):
        train, _, _ = self._run()
        assert "BUREAU_INQUIRIES_TOTAL" in train.columns

    def test_no_object_columns_remain(self):
        train, val, test = self._run()
        assert train.select_dtypes(include="object").empty, "Object columns remain in train"
        assert val.select_dtypes(include="object").empty,   "Object columns remain in val"
        assert test.select_dtypes(include="object").empty,  "Object columns remain in test"

    def test_target_preserved_in_labeled_splits(self):
        train, val, test = self._run()
        assert "TARGET" in train.columns
        assert "TARGET" in val.columns
        assert "TARGET" not in test.columns  # test has no labels

    def test_val_and_test_columns_subset_of_train(self):
        """Val and test must never have columns that train does not (OHE unseen categories)."""
        train, val, test = self._run()
        extra_val  = set(val.columns)  - set(train.columns)
        extra_test = set(test.columns) - set(train.columns)
        assert extra_val  == set(), f"Val has extra columns: {extra_val}"
        assert extra_test == set(), f"Test has extra columns: {extra_test}"

    def test_test_has_no_target(self):
        _, _, test = self._run()
        assert "TARGET" not in test.columns


# select_features tests


class TestSelectFeatures:

    @pytest.fixture(autouse=True)
    def _features(self):
        self.train_f, self.val_f, self.test_f = create_features(
            _make_app_df(100, seed=42, has_target=True),
            _make_app_df(25,  seed=77, has_target=True),
            _make_app_df(20,  seed=88, has_target=False),
            PARAMS,
        )

    def test_x_train_has_correct_n_features(self):
        X_train, _, _, _, _, best_cols = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert len(best_cols) == PARAMS["n_features_to_select"]
        assert X_train.shape[1] == PARAMS["n_features_to_select"] + 1  # +1 for SK_ID_CURR

    def test_x_train_contains_id_column(self):
        X_train, _, X_val, _, X_test, _ = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert "SK_ID_CURR" in X_train.columns
        assert "SK_ID_CURR" in X_val.columns
        assert "SK_ID_CURR" in X_test.columns

    def test_x_val_has_same_columns_as_x_train(self):
        X_train, _, X_val, _, _, _ = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert list(X_train.columns) == list(X_val.columns)

    def test_x_test_has_same_columns_as_x_train(self):
        X_train, _, _, _, X_test, _ = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert list(X_train.columns) == list(X_test.columns)

    def test_y_train_shape(self):
        _, y_train, _, _, _, _ = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert y_train.shape == (100, 1)
        assert "TARGET" in y_train.columns

    def test_y_val_shape(self):
        _, _, _, y_val, _, _ = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert y_val.shape == (25, 1)
        assert "TARGET" in y_val.columns

    def test_no_y_test(self):
        """select_features returns X_test only - test has no ground-truth labels."""
        result = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert len(result) == 6  # X_train, y_train, X_val, y_val, X_test, best_cols

    def test_best_columns_list_length(self):
        _, _, _, _, _, best_cols = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert len(best_cols) == PARAMS["n_features_to_select"]

    def test_best_columns_no_target_or_id(self):
        _, _, _, _, _, best_cols = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert "TARGET"     not in best_cols
        assert "SK_ID_CURR" not in best_cols

    def test_x_train_no_nulls(self):
        X_train, _, _, _, _, _ = select_features(self.train_f, self.val_f, self.test_f, PARAMS)
        assert X_train.isnull().sum().sum() == 0



# select_features schema error tests


class TestSelectFeaturesSchemaError:

    def _base(self):
        return create_features(
            _make_app_df(100, seed=42, has_target=True),
            _make_app_df(25,  seed=77, has_target=True),
            _make_app_df(20,  seed=88, has_target=False),
            PARAMS,
        )

    def test_val_schema_mismatch_raises(self):
        train_f, val_f, test_f = self._base()
        # Strip all features from val - any RFE-selected column will be missing
        val_empty = val_f[["TARGET"]]
        with pytest.raises(ValueError, match="X_val missing columns"):
            select_features(train_f, val_empty, test_f, PARAMS)

    def test_test_schema_mismatch_raises(self):
        train_f, val_f, test_f = self._base()
        # Strip all features from test
        test_empty = pd.DataFrame(index=test_f.index)
        with pytest.raises(ValueError, match="X_test missing columns"):
            select_features(train_f, val_f, test_empty, PARAMS)



# Target encoding tests


PARAMS_TE = {**PARAMS, "target_encoding_cols": ["OCCUPATION_TYPE"]}


class TestTargetEncoding:

    def _run(self):
        return create_features(
            _make_app_df(80, seed=42, has_target=True),
            _make_app_df(20, seed=99, has_target=True),
            _make_app_df(15, seed=77, has_target=False),
            PARAMS_TE,
        )

    def test_te_column_created_in_train_and_val(self):
        train, val, _ = self._run()
        assert "OCCUPATION_TYPE_TE" in train.columns
        assert "OCCUPATION_TYPE_TE" in val.columns

    def test_original_column_dropped(self):
        train, val, test = self._run()
        assert "OCCUPATION_TYPE" not in train.columns
        assert "OCCUPATION_TYPE" not in val.columns
        assert "OCCUPATION_TYPE" not in test.columns

    def test_te_values_are_probabilities(self):
        train, _, _ = self._run()
        assert (train["OCCUPATION_TYPE_TE"] >= 0).all()
        assert (train["OCCUPATION_TYPE_TE"] <= 1).all()

    def test_no_object_columns_after_te(self):
        train, val, test = self._run()
        assert train.select_dtypes(include="object").empty
        assert val.select_dtypes(include="object").empty
        assert test.select_dtypes(include="object").empty



# select_features clamp / validation tests


class TestSelectFeaturesClamp:

    def _features(self):
        return create_features(
            _make_app_df(30, seed=1, has_target=True),
            _make_app_df(10, seed=2, has_target=True),
            _make_app_df(8,  seed=3, has_target=False),
            PARAMS,
        )

    def test_n_select_clamped_when_exceeds_available(self):
        train_f, val_f, test_f = self._features()
        # Exclude TARGET and SK_ID_CURR 
        n_available = train_f.shape[1] - 2
        params_excessive = {**PARAMS, "n_features_to_select": n_available + 50}

        X_train, _, _, _, _, best_cols = select_features(train_f, val_f, test_f, params_excessive)

        assert len(best_cols) == n_available
        assert X_train.shape[1] == n_available + 1  # +1 for SK_ID_CURR

    def test_n_select_below_one_raises(self):
        train_f, val_f, test_f = self._features()
        params_invalid = {**PARAMS, "n_features_to_select": 0}
        with pytest.raises(ValueError, match="n_features_to_select"):
            select_features(train_f, val_f, test_f, params_invalid)
