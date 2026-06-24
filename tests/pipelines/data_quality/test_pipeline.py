"""
Tests for the 'data_quality' pipeline.

These tests load the *real* conf/base/parameters_data_quality.yml instead of
a hand-typed copy, so they always exercise the exact rules `kedro run` uses.
If someone edits the YAML and breaks a rule (typo in a column name, a
min_value above a max_value, etc.) these tests fail -- they can't drift out
of sync with production config the way a hardcoded fixture could.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from home_credit_mlops.pipelines.data_quality.nodes import (
    run_data_quality,
    run_table_quality,
)

CONF_BASE = Path(__file__).resolve().parents[3] / "conf" / "base"
PARAMS_PATH = CONF_BASE / "parameters_data_quality.yml"
PARAMS = yaml.safe_load(PARAMS_PATH.read_text())

NUMERICAL_RULES = PARAMS["numerical_rules"]
CATEGORICAL_RULES = PARAMS["categorical_rules"]
TARGET_RULES = PARAMS["target_rules"]
UNIQUE_COLUMNS = PARAMS["unique_columns"]

SECONDARY_TABLES = [
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "installments_payments",
    "credit_card_balance",
]

N_ROWS = 5


def _valid_dataframe() -> pd.DataFrame:
    """A dataframe with every column referenced in parameters_data_quality.yml,
    each set to a value that satisfies its rule (using the boundary value
    itself where there is one, since GX's between-expectation is inclusive)."""
    data = {}

    for column, bounds in NUMERICAL_RULES.items():
        min_value = bounds.get("min_value")
        max_value = bounds.get("max_value")
        value = min_value if min_value is not None else max_value
        data[column] = [value] * N_ROWS

    for column, value_set in CATEGORICAL_RULES.items():
        data[column] = [value_set[0]] * N_ROWS

    for column, value_set in TARGET_RULES.items():
        data[column] = [value_set[i % len(value_set)] for i in range(N_ROWS)]

    for column in UNIQUE_COLUMNS:
        data[column] = list(range(100000, 100000 + N_ROWS))

    return pd.DataFrame(data)


def _valid_secondary_dataframe(table_name: str) -> pd.DataFrame:
    """Same idea as _valid_dataframe, generalized to any secondary table:
    every column referenced in that table's rule blocks is set to a value
    that satisfies its rule."""
    numerical_rules = PARAMS.get(f"{table_name}_numerical_rules", {}) or {}
    categorical_rules = PARAMS.get(f"{table_name}_categorical_rules", {}) or {}
    unique_columns = PARAMS.get(f"{table_name}_unique_columns", []) or []
    not_null_columns = PARAMS.get(f"{table_name}_not_null_columns", []) or []

    data = {}

    for column, bounds in numerical_rules.items():
        min_value = bounds.get("min_value")
        max_value = bounds.get("max_value")
        value = min_value if min_value is not None else max_value
        data[column] = [value] * N_ROWS

    for column, value_set in categorical_rules.items():
        data[column] = [value_set[0]] * N_ROWS

    for column in unique_columns:
        data[column] = list(range(100000, 100000 + N_ROWS))

    for column in not_null_columns:
        if column not in data:
            data[column] = [1] * N_ROWS

    return pd.DataFrame(data)


def _run(data: pd.DataFrame):
    return run_data_quality(
        data, NUMERICAL_RULES, CATEGORICAL_RULES, TARGET_RULES, UNIQUE_COLUMNS
    )


def _run_secondary(table_name: str, data: pd.DataFrame):
    return run_table_quality(
        data,
        numerical_rules=PARAMS.get(f"{table_name}_numerical_rules", {}),
        categorical_rules=PARAMS.get(f"{table_name}_categorical_rules", {}),
        unique_columns=PARAMS.get(f"{table_name}_unique_columns", []),
        not_null_columns=PARAMS.get(f"{table_name}_not_null_columns", []),
        table_name=table_name,
    )


class TestDataQualityPipeline:
    def test_valid_data_passes_and_is_returned_unchanged(self):
        data = _valid_dataframe()

        validated_data, report = _run(data)

        assert validated_data.equals(data)
        assert report["success"].all()

    def test_report_has_one_row_per_expectation(self):
        data = _valid_dataframe()
        # every target column gets both a value-set AND a not-null expectation
        n_expected = (
            len(NUMERICAL_RULES)
            + len(CATEGORICAL_RULES)
            + 2 * len(TARGET_RULES)
            + len(UNIQUE_COLUMNS)
        )

        _, report = _run(data)

        assert len(report) == n_expected
        assert set(report.columns) == {"expectation_type", "column", "success"}

    @pytest.mark.parametrize("column", list(NUMERICAL_RULES.keys()))
    def test_each_numerical_rule_rejects_a_violating_value(self, column):
        bounds = NUMERICAL_RULES[column]
        data = _valid_dataframe()
        if bounds.get("min_value") is not None:
            data.loc[0, column] = bounds["min_value"] - 1
        else:
            data.loc[0, column] = bounds["max_value"] + 1

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run(data)

    @pytest.mark.parametrize("column", list(CATEGORICAL_RULES.keys()))
    def test_each_categorical_rule_rejects_an_unknown_value(self, column):
        data = _valid_dataframe()
        data.loc[0, column] = "NOT_A_REAL_CATEGORY"

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run(data)

    def test_invalid_target_value_raises_and_halts(self):
        data = _valid_dataframe()
        data.loc[0, "TARGET"] = 99

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run(data)

    def test_null_target_raises_and_halts(self):
        data = _valid_dataframe()
        data.loc[0, "TARGET"] = None

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run(data)

    def test_ext_source_boundaries_0_and_1_are_valid(self):
        # EXT_SOURCE_2/3 are the only rules with both a min AND a max:
        # this is the one place an off-by-one (e.g. strict_min=True) would
        # silently reject perfectly valid scores at the edges of [0, 1].
        data = _valid_dataframe()
        data["EXT_SOURCE_2"] = [0, 1, 0, 1, 0]
        data["EXT_SOURCE_3"] = [0, 1, 0, 1, 0]

        _, report = _run(data)

        assert report["success"].all()

    def test_multiple_simultaneous_failures_are_all_reported(self):
        data = _valid_dataframe()
        data.loc[0, "CODE_GENDER"] = "NOT_A_REAL_CATEGORY"
        data.loc[0, "AMT_INCOME_TOTAL"] = -1

        with pytest.raises(ValueError, match=r"2 expectation\(s\) failed"):
            _run(data)

    def test_sk_id_curr_uniqueness_rule_rejects_duplicate(self):
        # EDA cell-61: "Expect: SK_ID_CURR unique per row"
        data = _valid_dataframe()
        data.loc[1, "SK_ID_CURR"] = data.loc[0, "SK_ID_CURR"]

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run(data)


class TestSecondaryTableQuality:
    """The 6 secondary raw tables (bureau, bureau_balance,
    previous_application, pos_cash_balance, installments_payments,
    credit_card_balance) are validated by the same generic run_table_quality
    node. These tests exercise every rule for every table from the real YAML,
    exactly like TestDataQualityPipeline does for application_train."""

    @pytest.mark.parametrize("table_name", SECONDARY_TABLES)
    def test_valid_data_passes(self, table_name):
        data = _valid_secondary_dataframe(table_name)

        _, report = _run_secondary(table_name, data)

        assert report["success"].all()

    @pytest.mark.parametrize("table_name", SECONDARY_TABLES)
    def test_report_has_one_row_per_expectation(self, table_name):
        numerical_rules = PARAMS.get(f"{table_name}_numerical_rules", {}) or {}
        categorical_rules = PARAMS.get(f"{table_name}_categorical_rules", {}) or {}
        unique_columns = PARAMS.get(f"{table_name}_unique_columns", []) or []
        not_null_columns = PARAMS.get(f"{table_name}_not_null_columns", []) or []
        n_expected = (
            len(numerical_rules)
            + len(categorical_rules)
            + len(unique_columns)
            + len(not_null_columns)
        )

        data = _valid_secondary_dataframe(table_name)
        _, report = _run_secondary(table_name, data)

        assert len(report) == n_expected

    @pytest.mark.parametrize(
        "table_name,column",
        [
            (table_name, column)
            for table_name in SECONDARY_TABLES
            for column in (PARAMS.get(f"{table_name}_numerical_rules", {}) or {})
        ],
    )
    def test_each_numerical_rule_rejects_a_violating_value(self, table_name, column):
        bounds = PARAMS[f"{table_name}_numerical_rules"][column]
        data = _valid_secondary_dataframe(table_name)
        if bounds.get("min_value") is not None:
            data.loc[0, column] = bounds["min_value"] - 1
        else:
            data.loc[0, column] = bounds["max_value"] + 1

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run_secondary(table_name, data)

    @pytest.mark.parametrize(
        "table_name,column",
        [
            (table_name, column)
            for table_name in SECONDARY_TABLES
            for column in (PARAMS.get(f"{table_name}_categorical_rules", {}) or {})
        ],
    )
    def test_each_categorical_rule_rejects_an_unknown_value(self, table_name, column):
        data = _valid_secondary_dataframe(table_name)
        data.loc[0, column] = "NOT_A_REAL_CATEGORY"

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run_secondary(table_name, data)

    @pytest.mark.parametrize(
        "table_name", [t for t in SECONDARY_TABLES if PARAMS.get(f"{t}_unique_columns")]
    )
    def test_unique_columns_rule_rejects_duplicate(self, table_name):
        unique_column = PARAMS[f"{table_name}_unique_columns"][0]
        data = _valid_secondary_dataframe(table_name)
        data.loc[1, unique_column] = data.loc[0, unique_column]

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run_secondary(table_name, data)

    @pytest.mark.parametrize(
        "table_name",
        [t for t in SECONDARY_TABLES if PARAMS.get(f"{t}_not_null_columns")],
    )
    def test_not_null_columns_rule_rejects_null(self, table_name):
        not_null_column = PARAMS[f"{table_name}_not_null_columns"][0]
        data = _valid_secondary_dataframe(table_name)
        data.loc[0, not_null_column] = None

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run_secondary(table_name, data)
