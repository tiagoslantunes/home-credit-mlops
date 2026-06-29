"""Pipeline of model_predict.

Reads:  X_val_data (features to score), production_model, production_columns,
        decision_threshold, ``params:model_predict``; optionally y_val_data.
Writes: predictions (CSV in data/07_model_output/), serving_metrics (JSON in
        data/08_reporting/) when labels are available.
"""

from kedro.pipeline import Pipeline, node

from .nodes import evaluate_predictions, predict


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=predict,
                inputs={
                    "features": "X_val_data",
                    "model": "production_model",
                    "decision_threshold": "decision_threshold",
                    "params": "params:model_predict",
                },
                outputs="predictions",
                name="predict",
                tags=["model_predict"],
            ),
            node(
                func=evaluate_predictions,
                inputs={
                    "predictions": "predictions",
                    "y_true": "y_val_data",
                    "decision_threshold": "decision_threshold",
                },
                outputs="serving_metrics",
                name="evaluate_predictions",
                tags=["model_predict"],
            ),
        ]
    )
