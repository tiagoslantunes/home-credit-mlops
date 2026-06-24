"""
This is a boilerplate pipeline 'data_quality'
generated using Kedro 1.4.0
"""

import logging

import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)


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
    evidence behind each rule). Generic across tables: callers compose
    table-specific concepts (e.g. application_train's TARGET) by merging them
    into these four primitives before calling this function."""
    numerical_rules = numerical_rules or {}
    categorical_rules = categorical_rules or {}
    unique_columns = unique_columns or []
    not_null_columns = not_null_columns or []

    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

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
                value_set=value_set,
            )
        )

    for column in unique_columns:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column=column))

    for column in not_null_columns:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))

    return suite


def _validate(
    data: pd.DataFrame,
    suite: gx.ExpectationSuite,
    context: gx.data_context.AbstractDataContext,
    table_name: str,
):
    data_source = context.data_sources.add_pandas(f"{table_name}_source")
    data_asset = data_source.add_dataframe_asset(f"{table_name}_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": data})
    return batch.validate(suite)


def _finalize(
    data: pd.DataFrame,
    results,
    table_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the report dataframe and gate the pipeline on failure.

    Raises:
        ValueError: if any expectation fails, halting the pipeline before
            any cleaning/training step can run on invalid data.
    """
    report = pd.DataFrame(
        [
            {
                "expectation_type": result.expectation_config.type,
                "column": result.expectation_config.kwargs.get("column", ""),
                "success": result.success,
            }
            for result in results.results
        ]
    )

    n_failed = int((~report["success"]).sum())
    logger.info(
        "Data quality validation for %s: %d/%d expectations passed",
        table_name,
        len(report) - n_failed,
        len(report),
    )

    if not results.success:
        failed = report[~report["success"]]
        raise ValueError(
            f"Data quality validation failed for {table_name}. "
            f"{n_failed} expectation(s) failed:\n{failed.to_string(index=False)}"
        )

    return data, report


def run_data_quality(
    data: pd.DataFrame,
    numerical_rules: dict,
    categorical_rules: dict,
    target_rules: dict,
    unique_columns: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate application_train against the EDA-derived Great Expectations
    rules and gate the pipeline on failure.

    Returns the (unchanged) validated dataframe alongside a report dataframe,
    so that downstream pipelines depend on the validated output rather than
    on the raw CSV directly -- this is what makes the gate binding across
    separate `kedro run --pipeline=...` invocations, not just within one.

    Raises:
        ValueError: if any expectation fails, halting the pipeline before
            any cleaning/training step can run on invalid data.
    """
    context = gx.get_context(mode="ephemeral")
    suite = _build_expectation_suite(
        context,
        "application_train_quality",
        numerical_rules=numerical_rules,
        categorical_rules={**categorical_rules, **target_rules},
        unique_columns=unique_columns,
        not_null_columns=list(target_rules.keys()),
    )
    results = _validate(data, suite, context, "application_train")
    return _finalize(data, results, "application_train")


def run_table_quality(  
    data: pd.DataFrame,
    numerical_rules: dict | None = None,
    categorical_rules: dict | None = None,
    unique_columns: list | None = None,
    not_null_columns: list | None = None,
    *,
    table_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate one of the secondary raw tables (bureau, bureau_balance,
    previous_application, pos_cash_balance, installments_payments,
    credit_card_balance) against EDA-derived rules and gate the pipeline on
    failure. None of these tables have a TARGET column, so this is a thinner,
    fully generic counterpart to run_data_quality.

    Raises:
        ValueError: if any expectation fails, halting the pipeline before
            any cleaning/training step can run on invalid data.
    """
    context = gx.get_context(mode="ephemeral")
    suite = _build_expectation_suite(
        context,
        f"{table_name}_quality",
        numerical_rules=numerical_rules,
        categorical_rules=categorical_rules,
        unique_columns=unique_columns,
        not_null_columns=not_null_columns,
    )
    results = _validate(data, suite, context, table_name)
    return _finalize(data, results, table_name)
