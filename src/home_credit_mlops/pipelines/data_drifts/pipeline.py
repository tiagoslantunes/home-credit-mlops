"""Pipeline of data_drifts.

Compares two feature matrices and decides whether the model's training
distribution still describes the data being scored.

Reads:  X_train_data (reference) and X_val_data (current); ``params:data_drifts``
Writes: drift_psi_metrics (CSV), drift_report_html (HTML), drift_alerts (JSON)
"""

from kedro.pipeline import Pipeline, node

from .nodes import compute_evidently_report, compute_psi_metrics, raise_drift_alerts


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=compute_psi_metrics,
                inputs={
                    "reference": "X_train_data",
                    "current": "X_val_data",
                    "params": "params:data_drifts",
                },
                outputs="drift_psi_metrics",
                name="compute_psi_metrics",
                tags=["data_drifts"],
            ),
            node(
                func=compute_evidently_report,
                inputs={
                    "reference": "X_train_data",
                    "current": "X_val_data",
                    "params": "params:data_drifts",
                },
                outputs=["drift_report_html", "drift_evidently_summary"],
                name="compute_evidently_report",
                tags=["data_drifts"],
            ),
            node(
                func=raise_drift_alerts,
                inputs={
                    "psi_metrics": "drift_psi_metrics",
                    "evidently_summary": "drift_evidently_summary",
                    "params": "params:data_drifts",
                },
                outputs="drift_alerts",
                name="raise_drift_alerts",
                tags=["data_drifts"],
            ),
        ]
    )
