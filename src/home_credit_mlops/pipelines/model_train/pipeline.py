"""
Pipeline de model_train.

Lê:   X_train_data, y_train_data   (data/05_model_input/)  — treino
      X_val_data, y_val_data        (data/05_model_input/)  — validação/avaliação
      best_params                   (data/06_models/best_params.json)
Gera: production_model             (data/06_models/production_model.pkl via MLflow)
      decision_threshold            (data/06_models/decision_threshold.json)
      model_metrics                 (data/08_reporting/production_model_metrics.json)
      shap_importance               (data/08_reporting/shap_importance.csv)
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    evaluate_and_log_model,
    generate_shap_explanations,
    select_decision_threshold,
    train_final_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_final_model,
                inputs={
                    "X_train": "X_train_data",
                    "y_train": "y_train_data",
                    "best_params": "best_params",
                    "params": "params:model_train",
                },
                outputs="production_model",
                name="train_final_model",
                tags=["model_train"],
            ),
            # Threshold selecionado ANTES da avaliação para que as métricas
            # threshold-dependentes reportadas correspondam ao corte de produção.
            node(
                func=select_decision_threshold,
                inputs={
                    "model": "production_model",
                    "X_validation": "X_val_data",
                    "y_validation": "y_val_data",
                    "params": "params:model_train",
                },
                outputs="decision_threshold",
                name="select_decision_threshold",
                tags=["model_train"],
            ),
            node(
                func=evaluate_and_log_model,
                inputs={
                    "model": "production_model",
                    "X_train": "X_train_data",
                    "y_train": "y_train_data",
                    "X_test": "X_val_data",
                    "y_test": "y_val_data",
                    "params": "params:model_train",
                    "decision_threshold": "decision_threshold",
                },
                outputs="model_metrics",
                name="evaluate_and_log_model",
                tags=["model_train"],
            ),
            node(
                func=generate_shap_explanations,
                inputs={
                    "model": "production_model",
                    "X_test": "X_val_data",
                    "params": "params:model_train",
                    "y_test": "y_val_data",
                },
                outputs="shap_importance",
                name="generate_shap_explanations",
                tags=["model_train"],
            ),
        ]
    )
