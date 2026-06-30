"""
This is a boilerplate pipeline 'data_quality'
generated using Kedro 1.4.0
"""

from kedro.pipeline import Pipeline, node  # noqa

from .nodes import check_quality_gate, validate_data, validate_test_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=validate_data,
                inputs=[
                    "application_train",
                    "params:numerical_rules",
                    "params:categorical_rules",
                    "params:target_rules",
                    "params:unique_columns",
                ],
                outputs=["application_train_checked", "data_quality_report"],
                name="validate_train_data_node",
            ),
            node(
                func=check_quality_gate,
                inputs=["application_train_checked", "data_quality_report"],
                outputs="application_train_validated",
                name="quality_gate_train_node",
            ),
            node(
                func=validate_test_data,
                inputs=[
                    "application_test",
                    "params:numerical_rules",
                    "params:categorical_rules",
                    "params:unique_columns",
                ],
                outputs=["application_test_checked", "data_quality_report_test"],
                name="validate_test_data_node",
            ),
            node(
                func=check_quality_gate,
                inputs=["application_test_checked", "data_quality_report_test"],
                outputs="application_test_validated",
                name="quality_gate_test_node",
            ),
        ]
    )
