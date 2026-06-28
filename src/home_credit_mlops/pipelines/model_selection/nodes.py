"""
Nodes para a pipeline de model_selection.

Compara cinco famílias de modelos — RandomForest, HistGradientBoosting,
LightGBM, GradientBoosting (legacy), e WOE + LogisticRegression scorecard —
via duas estratégias complementares (W2/W7):

1. ``run_gridsearch_with_mlflow``: GridSearchCV sobre grids por família
   definidos em ``parameters_model_selection.yml``, com MLflow autologging.
   GridSearchCV é a abordagem clássica vista nas aulas teóricas (W2).

2. ``run_optuna_search``: Pesquisa Bayesiana TPE sobre as mesmas famílias
   usando o espaço definido em :mod:`home_credit_mlops.modeling`.
   Optuna (W7) usa Tree-structured Parzen Estimator para convergir mais
   rapidamente do que Grid/Random search.

``choose_best_params`` seleciona a família com maior ROC-AUC em CV.
A construção dos estimadores é delegada a
:func:`home_credit_mlops.modeling.build_estimator` (DRY — partilhado
com model_train).  A comparação de cinco famílias segue Lessmann et al. (2015).
"""

from __future__ import annotations

import logging
from math import prod
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from home_credit_mlops.modeling import MODEL_TYPES, build_estimator, suggest_hyperparams

logger = logging.getLogger(__name__)

# Suprimir logging verboso do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _select_search_sample(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Retorna uma amostra estratificada para a pesquisa de hiperparâmetros."""
    y_arr = (
        y_train.values.ravel()
        if hasattr(y_train, "values")
        else np.array(y_train).ravel()
    )
    max_rows = params.get("max_search_rows")
    if not max_rows or len(X_train) <= int(max_rows):
        return X_train, y_arr

    values, counts = np.unique(y_arr, return_counts=True)
    stratify = y_arr if len(values) > 1 and counts.min() >= 2 else None
    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_arr,
        train_size=int(max_rows),
        random_state=params.get("random_state", 42),
        stratify=stratify,
    )
    logger.info(
        "Model selection: amostra %d de %d linhas (estratificada=%s).",
        len(X_sample),
        len(X_train),
        stratify is not None,
    )
    return X_sample, np.asarray(y_sample).ravel()


def _build_cv(params: dict[str, Any]) -> StratifiedKFold | RepeatedStratifiedKFold:
    """Constrói o CV splitter estratificado configurado."""
    folds = int(params.get("cv_folds", 5))
    repeats = int(params.get("cv_repeats", 1))
    random_state = params.get("random_state", 42)

    if repeats > 1:
        return RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=random_state,
        )

    return StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )


def _count_grid(param_grid: dict) -> int:
    return prod(len(v) for v in param_grid.values()) if param_grid else 1


def choose_best_params(
    gs_best_params: dict[str, Any],
    optuna_best_params: dict[str, Any],
) -> dict[str, Any]:
    """Escolhe o melhor resultado de pesquisa pelo CV ROC-AUC comparável.

    GridSearchCV (abordagem sistemática clássica, W2) e Optuna TPE
    (pesquisa Bayesiana eficiente, W7) são executados independentemente;
    este nó escolhe o vencedor pelo score de validação cruzada.
    """
    gs_score = float(gs_best_params.get("cv_score", -np.inf))
    optuna_score = float(optuna_best_params.get("cv_score", -np.inf))

    if not np.isfinite(gs_score) and not np.isfinite(optuna_score):
        logger.warning("Sem cv_score comparável; usando Optuna best_params.")
        return dict(optuna_best_params or gs_best_params)

    winner = dict(gs_best_params) if gs_score >= optuna_score else dict(optuna_best_params)

    logger.info(
        "Vencedor da seleção de modelos: %s (%s) com CV ROC-AUC %.4f.",
        winner.get("model_type", "unknown"),
        winner.get("search_strategy", "unknown"),
        winner.get("cv_score", float("nan")),
    )
    return winner


# ─────────────────────────────────────────────────────────────────────────────
# GridSearchCV com MLflow autologging (W2)
# ─────────────────────────────────────────────────────────────────────────────


def run_gridsearch_with_mlflow(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Executa GridSearchCV para cada família de modelos e regista no MLflow.

    GridSearchCV (scikit-learn, W2) realiza uma pesquisa exaustiva sobre o
    produto cartesiano dos valores de hiperparâmetros configurados.  MLflow
    autologging (W2) regista automaticamente parâmetros e métricas de cada run.

    Args:
        X_train: Features de treino (05_model_input/X_train.csv).
        y_train: Labels de treino (05_model_input/y_train.csv).
        params: Configuração de ``params:model_selection`` — ``models``,
            ``cv_folds``, ``scoring``, ``max_search_rows``, ``random_state``.

    Returns:
        Tuple (best_params_dict, best_model_name).  ``best_params_dict`` inclui
        os melhores hiperparâmetros mais ``model_type``, ``cv_score`` e
        ``search_strategy``.
    """
    X_search, y_arr = _select_search_sample(X_train, y_train, params)

    cv = _build_cv(params)
    scoring = params.get("scoring", "roc_auc")
    models_cfg = params.get("models", {})
    random_state = params.get("random_state", 42)

    overall_best_score = -np.inf
    overall_best_params: dict[str, Any] = {}
    overall_best_model_name = ""

    mlflow.sklearn.autolog(log_models=False, silent=True)

    for model_name, cfg in models_cfg.items():
        param_grid = cfg.get("param_grid", {})

        logger.info(
            "GridSearch: a iniciar %s com %d combinações de parâmetros.",
            model_name,
            _count_grid(param_grid),
        )

        with mlflow.start_run(
            run_name=f"gridsearch_{model_name}",
            nested=mlflow.active_run() is not None,
        ):
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("search_strategy", "GridSearchCV")

            base_estimator = build_estimator(model_name, {}, random_state=random_state)
            gs = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            gs.fit(X_search, y_arr)

            best_score = float(gs.best_score_)
            best_params_gs = gs.best_params_

            mlflow.log_metric("best_cv_roc_auc", best_score)
            mlflow.log_params(best_params_gs)
            logger.info(
                "GridSearch %s: melhor CV ROC-AUC = %.4f | params = %s",
                model_name,
                best_score,
                best_params_gs,
            )

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_params = {
                    **best_params_gs,
                    "model_type": model_name,
                    "cv_score": best_score,
                    "search_strategy": "GridSearchCV",
                }
                overall_best_model_name = model_name

    mlflow.sklearn.autolog(disable=True)

    logger.info(
        "GridSearch vencedor: %s com CV ROC-AUC = %.4f",
        overall_best_model_name,
        overall_best_score,
    )
    return overall_best_params, overall_best_model_name


# ─────────────────────────────────────────────────────────────────────────────
# Optuna Bayesian search com MLflowCallback (W7)
# ─────────────────────────────────────────────────────────────────────────────


def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Executa pesquisa Optuna TPE entre as famílias candidatas com logging MLflow.

    Optuna (W7) usa Tree-structured Parzen Estimator (TPE), um método Bayesiano
    que modela a distribuição de hiperparâmetros que maximizam a métrica objetivo.
    É mais eficiente que GridSearch em espaços de alta dimensão.  MLflowCallback
    regista cada trial automaticamente no MLflow (W2+W7 integração).

    Returns:
        Dict com os melhores hiperparâmetros incluindo ``model_type``,
        ``cv_score`` e ``search_strategy``.
    """
    try:
        from optuna_integration.mlflow import MLflowCallback
    except ImportError:  # legacy optuna
        from optuna.integration import MLflowCallback  # type: ignore[no-redef]

    X_search, y_arr = _select_search_sample(X_train, y_train, params)

    cv = _build_cv(params)
    scoring = params.get("scoring", "roc_auc")
    random_state = params.get("random_state", 42)
    optuna_cfg = params.get("optuna", {})
    n_trials: int = optuna_cfg.get("n_trials", 50)
    model_types: list[str] = (
        optuna_cfg.get("model_types")
        or list(params.get("models", {}).keys())
        or MODEL_TYPES
    )

    nest = mlflow.active_run() is not None
    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=scoring,
        create_experiment=False,
        mlflow_kwargs={"run_name": "optuna_trial", "nested": nest},
        tag_study_user_attrs=True,
    )

    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical("model_type", model_types)
        hyperparams = suggest_hyperparams(trial, model_type)
        model = build_estimator(model_type, hyperparams, random_state=random_state)
        scores = cross_val_score(
            model, X_search, y_arr, cv=cv, scoring=scoring, n_jobs=-1
        )
        return float(scores.mean())

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    logger.info(
        "Optuna: %d trials (TPE, seed=%s) sobre famílias %s.",
        n_trials,
        random_state,
        model_types,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

    best_trial = study.best_trial
    best_params = {
        **best_trial.params,
        "cv_score": float(best_trial.value),
        "search_strategy": "Optuna_TPE",
    }

    with mlflow.start_run(run_name="optuna_best_result", nested=nest):
        mlflow.log_metric("best_optuna_roc_auc", best_trial.value)
        mlflow.log_params(best_params)
        mlflow.set_tag("search_strategy", "Optuna_TPE")

    logger.info(
        "Optuna best: ROC-AUC = %.4f | modelo = %s | params = %s",
        best_trial.value,
        best_params.get("model_type"),
        {k: v for k, v in best_params.items() if k != "model_type"},
    )
    return best_params
