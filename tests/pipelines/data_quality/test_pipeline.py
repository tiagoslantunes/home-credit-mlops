"""
Tests for the 'data_quality' pipeline.

These tests load the *real* conf/base/parameters_data_quality.yml instead of
a hand-typed copy, so they always exercise the exact rules `kedro run` uses.
If someone edits the YAML and breaks a rule (typo in a column name, a
min_value above a max_value, etc.) these tests fail -- they can't drift out
of sync with production config the way a hardcoded fixture could.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from home_credit_mlops.pipelines.data_quality import nodes
from home_credit_mlops.pipelines.data_quality.nodes import (
    check_quality_gate,
    validate_data,
    validate_test_data,
)

CONF_BASE = Path(__file__).resolve().parents[3] / "conf" / "base"
PARAMS_PATH = CONF_BASE / "parameters_data_quality.yml"
PARAMS = yaml.safe_load(PARAMS_PATH.read_text())

NUMERICAL_RULES = PARAMS["numerical_rules"]
CATEGORICAL_RULES = PARAMS["categorical_rules"]
TARGET_RULES = PARAMS["target_rules"]
UNIQUE_COLUMNS = PARAMS["unique_columns"]

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


def _valid_test_dataframe() -> pd.DataFrame:
    """Same shape as _valid_dataframe but without TARGET, mirroring
    application_test (no label column to validate)."""
    return _valid_dataframe().drop(columns=list(TARGET_RULES.keys()))


def _run(data: pd.DataFrame):
    """Run both data_quality train nodes back to back, exactly like the
    pipeline does."""
    validated_data, report = validate_data(
        data, NUMERICAL_RULES, CATEGORICAL_RULES, TARGET_RULES, UNIQUE_COLUMNS
    )
    check_quality_gate(report)
    return validated_data, report


def _run_test(data: pd.DataFrame):
    """Run both data_quality test nodes back to back, exactly like the
    pipeline does."""
    validated_data, report = validate_test_data(data, NUMERICAL_RULES, CATEGORICAL_RULES, UNIQUE_COLUMNS)
    check_quality_gate(report)
    return validated_data, report


def _fake_expectation_result(expectation_type, column, success, **result_fields):
    """Stand-in for a GX ExpectationValidationResult, shaped just enough to
    satisfy what _build_report actually reads off it (.expectation_config.type,
    .expectation_config.kwargs["column"], .success, .result.get(...)) -- no
    Great Expectations object is ever constructed."""
    return SimpleNamespace(
        expectation_config=SimpleNamespace(type=expectation_type, kwargs={"column": column}),
        success=success,
        result=result_fields,
    )


def _fake_validation_results(*expectation_results):
    """Stand-in for a GX ValidationResult: _build_report only ever reads
    `.results` off it."""
    return SimpleNamespace(results=list(expectation_results))


class TestBuildReport:
    """_build_report is a pure function (GX results in, DataFrame out): no
    Great Expectations context, no filesystem, no real validation run."""

    def test_maps_every_result_field_into_its_own_column(self):
        element_count = 5
        unexpected_count = 1
        unexpected_percent = 20.0
        partial_unexpected_list = [-1]
        results = _fake_validation_results(
            _fake_expectation_result(
                "expect_column_values_to_be_between",
                "AMT_INCOME_TOTAL",
                False,
                element_count=element_count,
                unexpected_count=unexpected_count,
                unexpected_percent=unexpected_percent,
                observed_value=None,
                partial_unexpected_list=partial_unexpected_list,
            )
        )

        report = nodes._build_report(results, "application_train")

        assert len(report) == 1
        row = report.iloc[0]
        assert row["expectation_type"] == "expect_column_values_to_be_between"
        assert row["column"] == "AMT_INCOME_TOTAL"
        assert not row["success"]
        assert row["element_count"] == element_count
        assert row["unexpected_count"] == unexpected_count
        assert row["unexpected_percent"] == unexpected_percent
        assert row["partial_unexpected_list"] == partial_unexpected_list

    def test_fields_absent_from_an_aggregate_expectation_become_none(self):
        # expect_column_distinct_values_to_be_in_set is a column-aggregate
        # expectation: GX's real result dict for it has no element_count or
        # unexpected_percent key at all (see _build_report's docstring).
        # .get() on the fabricated dict must degrade to None, not KeyError.
        results = _fake_validation_results(
            _fake_expectation_result(
                "expect_column_distinct_values_to_be_in_set",
                "CODE_GENDER",
                True,
                unexpected_count=0,
                partial_unexpected_list=[],
            )
        )

        report = nodes._build_report(results, "application_train")

        row = report.iloc[0]
        assert row["element_count"] is None
        assert row["unexpected_percent"] is None
        assert row["observed_value"] is None

    def test_preserves_order_and_count_for_multiple_results(self):
        results = _fake_validation_results(
            _fake_expectation_result(
                "expect_column_values_to_be_between", "A", True, element_count=5, unexpected_count=0
            ),
            _fake_expectation_result(
                "expect_column_values_to_be_between", "B", False, element_count=5, unexpected_count=2
            ),
        )

        report = nodes._build_report(results, "application_train")

        assert list(report["column"]) == ["A", "B"]
        assert list(report["success"]) == [True, False]


class TestCheckQualityGate:
    """check_quality_gate takes a plain DataFrame and either raises or
    doesn't: no GX, no validate_data, no filesystem."""

    def test_does_not_raise_when_every_row_succeeded(self):
        report = pd.DataFrame({"success": [True, True, True]})

        check_quality_gate(report)  # should not raise

    def test_does_not_raise_on_an_empty_report(self):
        report = pd.DataFrame({"success": pd.Series(dtype=bool)})

        check_quality_gate(report)  # should not raise

    def test_raises_with_the_correct_failure_count(self):
        report = pd.DataFrame({"success": [True, False, False]})

        with pytest.raises(ValueError, match=r"2 expectation\(s\) failed"):
            check_quality_gate(report)

    def test_error_message_names_the_failing_column(self):
        report = pd.DataFrame(
            {
                "expectation_type": ["expect_column_values_to_be_between"],
                "column": ["AMT_INCOME_TOTAL"],
                "success": [False],
            }
        )

        with pytest.raises(ValueError, match="AMT_INCOME_TOTAL"):
            check_quality_gate(report)


class TestDataQualityPipeline:
    @pytest.fixture(autouse=True)
    def _isolated_gx_project(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nodes, "GX_PROJECT_ROOT", str(tmp_path))

    def test_valid_data_passes_and_is_returned_unchanged(self):
        data = _valid_dataframe()

        validated_data, report = _run(data)

        assert validated_data.equals(data)
        assert report["success"].all()

    def test_report_has_one_row_per_expectation(self):
        data = _valid_dataframe()
        n_expected = (
            len(NUMERICAL_RULES)
            + len(CATEGORICAL_RULES)
            + 2 * len(TARGET_RULES)
            + len(UNIQUE_COLUMNS)
        )

        _, report = _run(data)

        assert len(report) == n_expected
        assert set(report.columns) == {
            "expectation_type",
            "column",
            "success",
            "element_count",
            "unexpected_count",
            "unexpected_percent",
            "observed_value",
            "partial_unexpected_list",
        }

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

    def test_report_is_returned_even_when_validation_fails(self):
        data = _valid_dataframe()
        data.loc[0, "AMT_INCOME_TOTAL"] = -1

        _, report = validate_data(
            data, NUMERICAL_RULES, CATEGORICAL_RULES, TARGET_RULES, UNIQUE_COLUMNS
        )

        assert not report["success"].all()
        failed_row = report[report["column"] == "AMT_INCOME_TOTAL"].iloc[0]
        assert failed_row["unexpected_count"] == 1
        assert failed_row["element_count"] == N_ROWS

    def test_validates_application_test_without_target_rules(self):
        # application_test has no TARGET column: validate_test_data must
        # validate it without ever referencing target_rules.
        data = _valid_test_dataframe()

        validated_data, report = _run_test(data)

        assert validated_data.equals(data)
        assert report["success"].all()
        assert "TARGET" not in set(report["column"])

    def test_invalid_application_test_data_raises_and_halts(self):
        data = _valid_test_dataframe()
        data.loc[0, "AMT_INCOME_TOTAL"] = -1

        with pytest.raises(ValueError, match="Data quality validation failed"):
            _run_test(data)

    def test_train_and_test_validations_do_not_collide_in_the_same_gx_project(self):
        # Regression guard for the whole point of table_name: both
        # validations persist to the same GX_PROJECT_ROOT, so they must get
        # independent suites/validation definitions instead of one
        # overwriting the other's history.
        _, train_report = _run(_valid_dataframe())
        _, test_report = _run_test(_valid_test_dataframe())

        assert train_report["success"].all()
        assert test_report["success"].all()
        assert "TARGET" in set(train_report["column"])
        assert "TARGET" not in set(test_report["column"])
