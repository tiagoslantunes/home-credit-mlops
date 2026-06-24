"""
This is a boilerplate pipeline 'data_quality'
generated using Kedro 1.4.0
"""

from functools import partial, update_wrapper

from kedro.pipeline import Pipeline, node  # noqa

from .nodes import run_data_quality, run_table_quality

SECONDARY_TABLES = [
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "installments_payments",
    "credit_card_balance",
]


def _secondary_table_node(table_name: str) -> node:
    func = partial(run_table_quality, table_name=table_name)
    update_wrapper(func, run_table_quality)
    return node(
        func=func,
        inputs=[
            table_name,
            f"params:{table_name}_numerical_rules",
            f"params:{table_name}_categorical_rules",
            f"params:{table_name}_unique_columns",
            f"params:{table_name}_not_null_columns",
        ],
        outputs=[f"{table_name}_validated", f"{table_name}_quality_report"],
        name=f"data_quality_{table_name}_node",
    )


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_data_quality,
                inputs=[
                    "application_train",
                    "params:numerical_rules",
                    "params:categorical_rules",
                    "params:target_rules",
                    "params:unique_columns",
                ],
                outputs=["application_train_validated", "data_quality_report"],
                name="data_quality_node",
            ),
            *[_secondary_table_node(table_name) for table_name in SECONDARY_TABLES],
        ]
    )
