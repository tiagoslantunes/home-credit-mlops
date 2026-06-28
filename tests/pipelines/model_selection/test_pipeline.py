"""
Testes unitários para os nodes da pipeline model_selection.

MLflow configurado para usar diretório temporário local.
Optuna corre com n_trials=2 para manter os testes rápidos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import mlflow

from home_credit_mlops.pipelines.model_selection.nodes import (
    _build_cv,
    _select_search_sample,
    choose_best_params,
    run_gridsearch_with_mlflow,
    run_optuna_search,
)

# ─────────────────────────────────────────────────────────────────────────────
# Params mínimos para testes rápidos
# ─────────────────────────────────────────────────────────────────────────────

GS_PARAMS = {
    "cv_folds": 2,
    "scoring": "roc_auc",
    "random_state": 42,
    "models": {
        "RandomForestClassifier": {
            "param_grid": {
                "n_estimators": [10],
                "max_depth": [3],
            },
        },
        "GradientBoostingClassifier": {
            "param_grid": {
                "n_estimators": [10],
                "learning_rate": [0.1],
                "max_depth": [2],
            },
        },
    },
    "optuna": {
        "n_trials": 2,
        "model_types": ["RandomForestClassifier", "GradientBoostingClassifier"],
    },
}


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path):
    """Configura MLflow para usar diretório temporário em todos os testes."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test_model_selection")
    yield
    mlflow.end_run()


@pytest.fixture
def toy_data():
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=42, n_classes=2
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y_df = pd.DataFrame({"TARGET": y})
    return X_df, y_df


# ─────────────────────────────────────────────────────────────────────────────
# _select_search_sample
# ─────────────────────────────────────────────────────────────────────────────


class TestSelectSearchSample:
    def test_sample_respeita_max_rows_e_mantém_ambas_classes(self, toy_data):
        X, y = toy_data
        params = {**GS_PARAMS, "max_search_rows": 40}

        X_sample, y_sample = _select_search_sample(X, y, params)

        assert len(X_sample) == 40
        assert set(np.unique(y_sample)) == {0, 1}

    def test_sem_max_rows_devolve_dataset_completo(self, toy_data):
        X, y = toy_data
        params = {**GS_PARAMS}
        params.pop("max_search_rows", None)

        X_sample, y_sample = _select_search_sample(X, y, params)

        assert len(X_sample) == len(X)


# ─────────────────────────────────────────────────────────────────────────────
# _build_cv
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildCV:
    def test_repeated_cv_usa_todos_os_repeats(self, toy_data):
        X, y = toy_data
        cv = _build_cv({"cv_folds": 3, "cv_repeats": 2, "random_state": 42})
        assert len(list(cv.split(X, y.values.ravel()))) == 6

    def test_cv_simples_sem_repeats(self, toy_data):
        X, y = toy_data
        cv = _build_cv({"cv_folds": 4, "cv_repeats": 1, "random_state": 42})
        assert len(list(cv.split(X, y.values.ravel()))) == 4


# ─────────────────────────────────────────────────────────────────────────────
# choose_best_params
# ─────────────────────────────────────────────────────────────────────────────


class TestChooseBestParams:
    def test_prefere_gridsearch_quando_cv_score_maior(self):
        gs_best = {"model_type": "GradientBoostingClassifier", "n_estimators": 80,
                   "cv_score": 0.743, "search_strategy": "GridSearchCV"}
        optuna_best = {"model_type": "GradientBoostingClassifier", "n_estimators": 78,
                       "cv_score": 0.741, "search_strategy": "Optuna_TPE"}

        result = choose_best_params(gs_best, optuna_best)

        assert result["n_estimators"] == 80
        assert result["search_strategy"] == "GridSearchCV"

    def test_prefere_optuna_quando_cv_score_maior(self):
        gs_best = {"model_type": "RandomForestClassifier", "n_estimators": 100,
                   "cv_score": 0.730, "search_strategy": "GridSearchCV"}
        optuna_best = {"model_type": "GradientBoostingClassifier", "n_estimators": 90,
                       "cv_score": 0.745, "search_strategy": "Optuna_TPE"}

        result = choose_best_params(gs_best, optuna_best)

        assert result["n_estimators"] == 90
        assert result["search_strategy"] == "Optuna_TPE"

    def test_sem_scores_validos_nao_crasha(self):
        gs_best = {"model_type": "RandomForestClassifier", "cv_score": float("-inf")}
        optuna_best = {"model_type": "GradientBoostingClassifier", "cv_score": float("-inf")}

        result = choose_best_params(gs_best, optuna_best)

        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# run_gridsearch_with_mlflow
# ─────────────────────────────────────────────────────────────────────────────


class TestRunGridsearchWithMlflow:
    def test_devolve_best_params_com_model_type(self, toy_data):
        X, y = toy_data
        best_params, best_model_name = run_gridsearch_with_mlflow(X, y, GS_PARAMS)

        assert isinstance(best_params, dict)
        assert "model_type" in best_params
        assert best_model_name in ("RandomForestClassifier", "GradientBoostingClassifier")

    def test_cria_runs_mlflow_por_familia(self, toy_data):
        """Deve ser criado pelo menos um run MLflow por família de modelos."""
        X, y = toy_data
        run_gridsearch_with_mlflow(X, y, GS_PARAMS)

        client = mlflow.tracking.MlflowClient()
        exp = mlflow.get_experiment_by_name("test_model_selection")
        runs = client.search_runs([exp.experiment_id])

        assert len(runs) >= 2

    def test_best_params_tem_hiperparametros(self, toy_data):
        X, y = toy_data
        best_params, _ = run_gridsearch_with_mlflow(X, y, GS_PARAMS)

        param_keys = set(best_params.keys()) - {"model_type", "cv_score", "search_strategy"}
        assert len(param_keys) > 0


# ─────────────────────────────────────────────────────────────────────────────
# run_optuna_search
# ─────────────────────────────────────────────────────────────────────────────


class TestRunOptunaSearch:
    def test_devolve_dict_com_model_type(self, toy_data):
        X, y = toy_data
        best_params = run_optuna_search(X, y, GS_PARAMS)

        assert isinstance(best_params, dict)
        assert "model_type" in best_params

    def test_regista_trials_no_mlflow(self, toy_data):
        X, y = toy_data
        run_optuna_search(X, y, GS_PARAMS)

        client = mlflow.tracking.MlflowClient()
        exp = mlflow.get_experiment_by_name("test_model_selection")
        runs = client.search_runs([exp.experiment_id])

        assert len(runs) > 0

    def test_best_params_constroem_estimador_valido(self, toy_data):
        """Os params do Optuna devem produzir um estimador treinável."""
        from home_credit_mlops.modeling import MODEL_TYPES, build_estimator

        X, y = toy_data
        best_params = run_optuna_search(X, y, GS_PARAMS)

        assert best_params["model_type"] in MODEL_TYPES
        hp = {k: v for k, v in best_params.items()
              if k not in {"model_type", "cv_score", "search_strategy"}}
        estimator = build_estimator(best_params["model_type"], hp)
        estimator.fit(X, y.values.ravel())
        assert hasattr(estimator, "predict_proba")
