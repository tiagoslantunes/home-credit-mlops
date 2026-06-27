"""
Data Feature Engineering pipeline.

Order:
  1. create_features       — engineer features from all three cleaned splits
  2. select_features       — RFE on train; applied to validation & test
  3. to_feature_store      — upload to Hopsworks (no-op if API key absent)

Inputs  (from 03_primary via catalog):
  preprocessed_train_data, preprocessed_validation_data, preprocessed_test_data

Outputs (to catalog):
  04_feature   : application_train_features, application_validation_features,
                 application_test_features
  05_model_input: X_train_data, y_train_data, X_val_data, y_val_data, X_test_data
  06_models    : best_columns
  08_reporting : feature_store_metadata
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_features, select_features, to_feature_store


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_features,
                inputs=[
                    "preprocessed_train_data",
                    "preprocessed_validation_data",
                    "preprocessed_test_data",
                    "parameters",
                ],
                outputs=[
                    "application_train_features",
                    "application_validation_features",
                    "application_test_features",
                ],
                name="create_features_node",
            ),
            node(
                func=select_features,
                inputs=[
                    "application_train_features",
                    "application_validation_features",
                    "application_test_features",
                    "parameters",
                ],
                outputs=[
                    "X_train_data",
                    "y_train_data",
                    "X_val_data",
                    "y_val_data",
                    "X_test_data",
                    "best_columns",
                ],
                name="feature_selection_node",
            ),
            node(
                func=to_feature_store,
                inputs=[
                    "X_train_data",
                    "y_train_data",
                    "X_val_data",
                    "y_val_data",
                    "X_test_data",
                    "parameters",
                ],
                outputs="feature_store_metadata",
                name="feature_store_node",
            ),
        ]
    )
