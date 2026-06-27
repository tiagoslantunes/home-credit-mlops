"""Pipeline definition for 'data_split'."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                # Split the data-quality-VALIDATED total, so the GX gate binds
                # before any train/validation partition is created.
                inputs=["application_train_validated", "params:data_split"],
                outputs=["application_train_split", "application_validation_split"],
                name="split_application_train_node",
            ),
        ]
    )
