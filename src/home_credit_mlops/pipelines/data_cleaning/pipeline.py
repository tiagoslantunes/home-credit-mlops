"""Pipeline definition for 'data_cleaning'.

fit_cleaning learns the artifact on the train split only; apply_cleaning then
replays it identically on both the train and validation splits (and is reusable
for any future serving batch).
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import apply_cleaning, fit_cleaning


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fit_cleaning,
                inputs=[
                    "application_train_split",
                    "params:data_cleaning",
                    "params:numerical_rules",
                ],
                outputs="cleaning_params",
                name="fit_cleaning_node",
            ),
            node(
                func=apply_cleaning,
                inputs=["application_train_split", "cleaning_params", "params:data_cleaning"],
                outputs="application_train_cleaned",
                name="clean_train_node",
            ),
            node(
                func=apply_cleaning,
                inputs=["application_validation_split", "cleaning_params", "params:data_cleaning"],
                outputs="application_validation_cleaned",
                name="clean_validation_node",
            ),
            # application_test is the production/holdout batch: it is NOT split,
            # it is cleaned straight from raw using the SAME train-fit artifact
            # (no TARGET column, so it yields the feature columns only).
            node(
                func=apply_cleaning,
                inputs=["application_test", "cleaning_params", "params:data_cleaning"],
                outputs="application_test_cleaned",
                name="clean_test_node",
            ),
        ]
    )
