"""
This is a boilerplate pipeline 'data_quality'
generated using Kedro 1.4.0
"""

import logging

import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)

GX_PROJECT_ROOT = "data/08_reporting"


def _get_context() -> gx.data_context.AbstractDataContext:
    return gx.get_context(project_root_dir=GX_PROJECT_ROOT)


def _build_expectation_suite(
    context: gx.data_context.AbstractDataContext,
    suite_name: str,
    numerical_rules: dict | None = None,
    categorical_rules: dict | None = None,
    unique_columns: list | None = None,
    not_null_columns: list | None = None,
) -> gx.ExpectationSuite:
    """Build a Great Expectations suite from the EDA-derived rules in
    conf/base/parameters_data_quality.yml (see notebooks/01_eda.ipynb for the
    evidence behind each rule)."""
    numerical_rules = numerical_rules or {}
    categorical_rules = categorical_rules or {}
    unique_columns = unique_columns or []
    not_null_columns = not_null_columns or []

    # add_or_update (not add) because the suite now lives in a persistent
    # context: on every run after the first, a suite with this name already
    # exists on disk. add_or_update replaces its contents wholesale, so a
    # rule removed from the YAML also disappears from the stored suite
    # instead of lingering from a previous run.
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add_or_update(suite)

    for column, bounds in numerical_rules.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=bounds.get("min_value"),
                max_value=bounds.get("max_value"),
            )
        )

    for column, value_set in categorical_rules.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                column=column,
                value_set=[v.upper() if isinstance(v, str) else v for v in value_set],
            )
        )

    for column in unique_columns:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column=column))

    for column in not_null_columns:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))

    return context.suites.add_or_update(suite)


def _casefold_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Upper-case the given string columns for validation purposes only, so
    categorical rules are case-insensitive. The dataframe returned to the
    caller of validate_data keeps the original casing -- only the copy
    that gets validated here is normalized."""
    data = data.copy()
    for column in columns:
        if column in data.columns and data[column].dtype == object:
            data[column] = data[column].str.upper()
    return data


def _get_or_add_asset(data_source, asset_name: str):
    try:
        return data_source.get_asset(asset_name)
    except Exception:
        return data_source.add_dataframe_asset(asset_name)


def _get_or_add_batch_definition(data_asset, batch_definition_name: str):
    try:
        return data_asset.get_batch_definition(batch_definition_name)
    except Exception:
        return data_asset.add_batch_definition_whole_dataframe(batch_definition_name)


def _validate(
    data: pd.DataFrame,
    suite: gx.ExpectationSuite,
    context: gx.data_context.AbstractDataContext,
    table_name: str,
):
    # A named, reusable ValidationDefinition (instead of a bare
    # batch.validate(suite) call) ties the data asset to the suite under a
    # stable name. That stability is what lets Data Docs group every run's
    # results together as history for the same check, rather than each run
    # leaving an orphaned, unrelated result.
    data_source = context.data_sources.add_or_update_pandas(f"{table_name}_source")
    data_asset = _get_or_add_asset(data_source, f"{table_name}_asset")
    batch_definition = _get_or_add_batch_definition(data_asset, "batch")
    validation_definition = context.validation_definitions.add_or_update(
        gx.core.validation_definition.ValidationDefinition(
            name=f"{table_name}_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    return validation_definition.run(batch_parameters={"dataframe": data})


def _build_report(results, table_name: str) -> pd.DataFrame:
    """Turn GX's validation results into a report with enough detail to act
    on a failure without re-running anything: how many rows were checked,
    how many violated the rule, what share that is, and a sample of the
    actual offending values. expect_column_distinct_values_to_be_in_set is a
    column-aggregate expectation rather than a row-level one, so it has no
    element_count/unexpected_percent -- those stay empty for categorical
    rules, which is expected, not a bug.
    """
    rows = []
    for result in results.results:
        result_detail = result.result
        rows.append(
            {
                "expectation_type": result.expectation_config.type,
                "column": result.expectation_config.kwargs.get("column", ""),
                "success": result.success,
                "element_count": result_detail.get("element_count"),
                "unexpected_count": result_detail.get("unexpected_count"),
                "unexpected_percent": result_detail.get("unexpected_percent"),
                "observed_value": result_detail.get("observed_value"),
                "partial_unexpected_list": result_detail.get("partial_unexpected_list"),
            }
        )
    report = pd.DataFrame(rows)

    n_failed = int((~report["success"]).sum())
    logger.info(
        "Data quality validation for %s: %d/%d expectations passed",
        table_name,
        len(report) - n_failed,
        len(report),
    )

    return report


def _run_validation(
    data: pd.DataFrame,
    table_name: str,
    numerical_rules: dict,
    categorical_rules: dict,
    target_rules: dict | None = None,
    unique_columns: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared engine behind validate_data/validate_test_data: build the
    suite, run it against `data`, build the report. table_name drives the
    GX suite/validation-definition names (so e.g. application_train and
    application_test get independent suites in the same persistent GX
    project instead of overwriting each other) and the report's log line.

    Always returns normally, even when expectations fail -- gating is the
    separate `check_quality_gate` node on purpose. A Kedro node only writes
    its outputs to the catalog after it returns; a node that built the
    report and raised in the same call would never persist that report on
    failure, which is exactly the moment it's needed for investigation.

    Returns the (unchanged) validated dataframe alongside a report dataframe,
    so that downstream pipelines depend on the validated output rather than
    on the raw CSV directly -- this is what makes the gate binding across
    separate `kedro run --pipeline=...` invocations, not just within one.
    """
    target_rules = target_rules or {}
    all_categorical_rules = {**categorical_rules, **target_rules}
    context = _get_context()
    suite = _build_expectation_suite(
        context,
        f"{table_name}_quality",
        numerical_rules=numerical_rules,
        categorical_rules=all_categorical_rules,
        unique_columns=unique_columns,
        not_null_columns=list(target_rules.keys()),
    )
    validation_data = _casefold_columns(data, list(all_categorical_rules.keys()))
    results = _validate(validation_data, suite, context, table_name)
    context.build_data_docs()
    report = _build_report(results, table_name)
    return data, report


def validate_data(
    data: pd.DataFrame,
    numerical_rules: dict,
    categorical_rules: dict,
    target_rules: dict,
    unique_columns: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate application_train against the EDA-derived Great Expectations
    rules and build a report. See _run_validation for the shared mechanics."""
    return _run_validation(
        data, "application_train", numerical_rules, categorical_rules, target_rules, unique_columns
    )


def validate_test_data(
    data: pd.DataFrame,
    numerical_rules: dict,
    categorical_rules: dict,
    unique_columns: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate application_test the same way as validate_data, minus
    target_rules: application_test has no TARGET column to check. See
    _run_validation for the shared mechanics."""
    return _run_validation(
        data, "application_test", numerical_rules, categorical_rules, unique_columns=unique_columns
    )


def check_quality_gate(data: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """Halt the pipeline if any data quality expectation failed.

    Raises:
        ValueError: if any expectation in `report` failed.
    """
    failed = report[~report["success"]]
    if not failed.empty:
        raise ValueError(
            f"Data quality validation failed. {len(failed)} expectation(s) "
            f"failed:\n{failed.to_string(index=False)}"
        )
    return data
