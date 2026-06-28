"""
Pipeline de model_selection.

Lê:   X_train_data, y_train_data  (data/05_model_input/)
Gera: best_params                 (data/06_models/best_params.json)
Mem:  gs_best_params, optuna_best_params, best_model_name  (MemoryDataset)

GridSearchCV (W2) e Optuna TPE (W7) correm independentemente sobre o
conjunto de treino; um nó final compara os CV ROC-AUC e persiste o vencedor.
"""

from kedro.pipeline import Pipeline, node

from .nodes import choose_best_params, run_gridsearch_with_mlflow, run_optuna_search


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_gridsearch_with_mlflow,
                inputs={
                    "X_train": "X_train_data",
                    "y_train": "y_train_data",
                    "params": "params:model_selection",
                },
                outputs=["gs_best_params", "best_model_name"],
                name="run_gridsearch_with_mlflow",
                tags=["model_selection"],
            ),
            node(
                func=run_optuna_search,
                inputs={
                    "X_train": "X_train_data",
                    "y_train": "y_train_data",
                    "params": "params:model_selection",
                },
                outputs="optuna_best_params",
                name="run_optuna_search",
                tags=["model_selection"],
            ),
            node(
                func=choose_best_params,
                inputs={
                    "gs_best_params": "gs_best_params",
                    "optuna_best_params": "optuna_best_params",
                },
                outputs="best_params",
                name="choose_best_params",
                tags=["model_selection"],
            ),
        ]
    )
