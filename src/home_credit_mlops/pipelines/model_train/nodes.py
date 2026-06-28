"""
Nodes da pipeline model_train.

Nodes:
1. ``train_final_model``         — treina o modelo vencedor da seleção (W2/W7)
   com calibração de probabilidade opcional (isotonic regression).
2. ``select_decision_threshold`` — escolhe o threshold de produção via F1,
   balanced accuracy ou matriz de custo assimétrica (Verbraken et al. 2014),
   adequada ao problema de risco de crédito onde falsos negativos custam mais.
3. ``evaluate_and_log_model``    — reporta o painel de métricas de credit
   scoring (ROC-AUC, Gini, KS, PR-AUC, Brier, log-loss) e regista tudo no
   MLflow (W2): parâmetros, métricas, artefactos (plots), e regista o modelo
   no MLflow Model Registry.
4. ``generate_shap_explanations`` — explicabilidade global com SHAP (W7):
   TreeExplainer para árvores, LinearExplainer para o scorecard WOE,
   fallback permutation importance para outros modelos.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from home_credit_mlops.modeling import build_estimator

matplotlib.use("Agg")  # backend não-interativo para ambientes de servidor

logger = logging.getLogger(__name__)

_METADATA_KEYS = {"model_type", "cv_score", "search_strategy"}
_TREE_TYPES = (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)


# ─────────────────────────────────────────────────────────────────────────────
# Treino (+ calibração opcional)
# ─────────────────────────────────────────────────────────────────────────────


def _strip_metadata(best_params: dict[str, Any]) -> dict[str, Any]:
    """Remove chaves de bookkeeping da seleção, deixando só hiperparâmetros."""
    return {k: v for k, v in best_params.items() if k not in _METADATA_KEYS}


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    best_params: dict[str, Any],
    params: dict[str, Any],
) -> Any:
    """Treina (e opcionalmente calibra) o modelo de produção.

    A família vencedora em ``best_params['model_type']`` é construída pela
    factory partilhada ``build_estimator`` (DRY com model_selection).  Quando
    ``params['calibrate']`` é true, o estimador é envolto em
    :class:`~sklearn.calibration.CalibratedClassifierCV` (isotonic por omissão)
    para que a probabilidade prevista seja uma verdadeira PD (Probability of
    Default).

    Raises:
        KeyError: Se ``best_params`` não tiver a chave ``model_type``.
    """
    if "model_type" not in best_params:
        raise KeyError("'model_type' em falta no dict best_params.")

    model_type = best_params["model_type"]
    hyperparams = _strip_metadata(best_params)
    y_arr = y_train.values.ravel()

    estimator = build_estimator(
        model_type,
        hyperparams,
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 42),
    )

    if params.get("calibrate", False):
        method = params.get("calibration_method", "isotonic")
        cv = int(params.get("calibration_cv", 3))
        logger.info(
            "Treino de %s com calibração de probabilidade %s (cv=%d) em %d x %d.",
            model_type, method, cv, *X_train.shape,
        )
        model: Any = CalibratedClassifierCV(estimator, method=method, cv=cv)
        model.fit(X_train, y_arr)
    else:
        logger.info(
            "Treino de %s em %d amostras x %d features.", model_type, *X_train.shape
        )
        estimator.fit(X_train, y_arr)
        model = estimator

    logger.info("Treino concluído (%s).", type(model).__name__)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Decision threshold
# ─────────────────────────────────────────────────────────────────────────────


def select_decision_threshold(
    model: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, float | str]:
    """Seleciona e persiste um threshold de probabilidade para produção.

    Estratégias (``params['threshold_strategy']``): ``f1_positive`` (default),
    ``f1_macro``, ``balanced_accuracy`` ou ``cost``.  A estratégia ``cost``
    minimiza o custo esperado de má-classificação com matriz assimétrica
    (``cost_false_negative`` >> ``cost_false_positive``), que é o objetivo
    adequado ao risco de crédito (Verbraken et al. 2014): um defaulter não
    detectado custa muito mais do que rejeitar um bom cliente.

    O threshold é escolhido num holdout estratificado do conjunto de validação
    para evitar viés otimista.
    """
    y_arr = y_validation.values.ravel()
    strategy = params.get("threshold_strategy", "f1_positive")
    validation_fraction = float(params.get("threshold_validation_fraction", 0.2))
    random_state = params.get("random_state", 42)

    score_model: Any = model
    X_score = X_validation
    y_score = y_arr
    values, counts = np.unique(y_arr, return_counts=True)

    if (
        0.0 < validation_fraction < 0.5
        and len(values) > 1
        and counts.min() >= 2
        and len(X_validation) >= 20
    ):
        X_fit, X_score, y_fit, y_score = train_test_split(
            X_validation,
            y_arr,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=y_arr,
        )
        score_model = clone(model)
        score_model.fit(X_fit, y_fit)
        source = "train_holdout"
    else:
        source = "training_predictions"

    proba = score_model.predict_proba(X_score)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_score, proba)

    cost_fn = float(params.get("cost_false_negative", 10.0))
    cost_fp = float(params.get("cost_false_positive", 1.0))

    if len(thresholds) == 0:
        threshold = 0.5
    else:
        min_threshold = float(params.get("threshold_min", 0.0))
        max_threshold = float(params.get("threshold_max", 1.0))
        threshold_mask = (thresholds >= min_threshold) & (thresholds <= max_threshold)
        if not threshold_mask.any():
            threshold_mask = np.ones_like(thresholds, dtype=bool)
        candidate_thresholds = thresholds[threshold_mask]

        if strategy == "f1_macro":
            scores = np.array(
                [f1_score(y_score, proba >= t, average="macro") for t in candidate_thresholds]
            )
        elif strategy == "balanced_accuracy":
            scores = np.array(
                [balanced_accuracy_score(y_score, proba >= t) for t in candidate_thresholds]
            )
        elif strategy == "cost":
            expected_costs = []
            for t in candidate_thresholds:
                pred = (proba >= t).astype(int)
                fp = float(np.sum((pred == 1) & (y_score == 0)))
                fn = float(np.sum((pred == 0) & (y_score == 1)))
                expected_costs.append(cost_fp * fp + cost_fn * fn)
            scores = -np.asarray(expected_costs)  # maximizar o negativo do custo
        else:
            strategy = "f1_positive"
            candidate_precision = precision[:-1][threshold_mask]
            candidate_recall = recall[:-1][threshold_mask]
            scores = (
                2
                * candidate_precision
                * candidate_recall
                / (candidate_precision + candidate_recall + 1e-12)
            )

        scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        threshold = (
            float(candidate_thresholds[int(np.argmax(scores))])
            if np.isfinite(scores).any()
            else 0.5
        )

    pred = (proba >= threshold).astype(int)
    fp = float(np.sum((pred == 1) & (y_score == 0)))
    fn = float(np.sum((pred == 0) & (y_score == 1)))
    artifact: dict[str, float | str] = {
        "threshold": float(threshold),
        "strategy": strategy,
        "source": source,
        "f1_positive": float(f1_score(y_score, pred, zero_division=0)),
        "f1_macro": float(f1_score(y_score, pred, average="macro")),
        "precision_positive": float(precision_score(y_score, pred, zero_division=0)),
        "recall_positive": float(recall_score(y_score, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_score, pred)),
        "expected_cost": float(cost_fp * fp + cost_fn * fn),
    }
    logger.info("Threshold de decisão selecionado: %s", artifact)
    return artifact


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação e registo no MLflow (W2)
# ─────────────────────────────────────────────────────────────────────────────


def _ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Estatística Kolmogorov-Smirnov = max(TPR - FPR)."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def evaluate_and_log_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    params: dict[str, Any],
    decision_threshold: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Avalia o modelo no painel de métricas de credit scoring e regista no MLflow.

    Métricas de ranking (threshold-free): ROC-AUC, Gini = 2·AUC−1, KS, PR-AUC,
    Brier score, log-loss.  Métricas dependentes de threshold (F1, precision,
    recall, balanced accuracy, custo esperado) são calculadas no threshold de
    produção selecionado — não no naive 0.5 — para que os números reportados
    correspondam ao que a produção vai fazer.

    Artefactos MLflow (W2): confusion matrix, calibration curve, PR curve;
    o modelo é logado e registado no MLflow Model Registry.
    """
    y_train_arr = y_train.values.ravel()
    y_test_arr = y_test.values.ravel()

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    threshold = 0.5
    if decision_threshold and "threshold" in decision_threshold:
        threshold = float(decision_threshold["threshold"])
    test_pred = (test_proba >= threshold).astype(int)

    cost_fn = float(params.get("cost_false_negative", 10.0))
    cost_fp = float(params.get("cost_false_positive", 1.0))
    fp = float(np.sum((test_pred == 1) & (y_test_arr == 0)))
    fn = float(np.sum((test_pred == 0) & (y_test_arr == 1)))

    test_auc = float(roc_auc_score(y_test_arr, test_proba))
    metrics = {
        "train_roc_auc": float(roc_auc_score(y_train_arr, train_proba)),
        "test_roc_auc": test_auc,
        "test_gini": 2.0 * test_auc - 1.0,
        "test_ks": _ks_statistic(y_test_arr, test_proba),
        "test_pr_auc": float(average_precision_score(y_test_arr, test_proba)),
        "test_brier": float(brier_score_loss(y_test_arr, test_proba)),
        "test_log_loss": float(log_loss(y_test_arr, test_proba)),
        "decision_threshold": float(threshold),
        "test_f1_positive": float(f1_score(y_test_arr, test_pred, zero_division=0)),
        "test_f1_macro": float(f1_score(y_test_arr, test_pred, average="macro")),
        "test_precision_positive": float(
            precision_score(y_test_arr, test_pred, zero_division=0)
        ),
        "test_recall_positive": float(
            recall_score(y_test_arr, test_pred, zero_division=0)
        ),
        "test_precision_macro": float(
            precision_score(y_test_arr, test_pred, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_test_arr, test_pred, average="macro", zero_division=0)
        ),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test_arr, test_pred)),
        "test_expected_cost": cost_fp * fp + cost_fn * fn,
        "test_cost_per_decision": (cost_fp * fp + cost_fn * fn) / max(len(y_test_arr), 1),
    }

    logger.info("Métricas: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    with mlflow.start_run(
        run_name="production_model_evaluation",
        nested=mlflow.active_run() is not None,
    ) as run:
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_class", type(model).__name__)

        _plot_confusion(y_test_arr, test_pred, params)
        _plot_calibration(y_test_arr, test_proba, params)
        _plot_pr_curve(y_test_arr, test_proba, params)

        registered_name = params.get(
            "mlflow_registered_model_name", "home_credit_production_model"
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="production_model",
            registered_model_name=registered_name,
        )
        try:
            version = getattr(model_info, "registered_model_version", None)
            if version is not None:
                mlflow.tracking.MlflowClient().set_registered_model_alias(
                    registered_name, "production", str(version)
                )
                logger.info(
                    "Modelo %s v%s promovido ao alias 'production'.",
                    registered_name, version,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Não foi possível definir alias 'production': %s", exc)
        logger.info(
            "Modelo registado como '%s' (run_id=%s).",
            registered_name, run.info.run_id,
        )

    return metrics


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, params: dict[str, Any]) -> None:
    path = params.get("confusion_matrix_path", "data/08_reporting/confusion_matrix.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (production threshold)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")


def _plot_calibration(y_true: np.ndarray, proba: np.ndarray, params: dict[str, Any]) -> None:
    path = params.get("calibration_curve_path", "data/08_reporting/calibration_curve.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration curve (validation)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")


def _plot_pr_curve(y_true: np.ndarray, proba: np.ndarray, params: dict[str, Any]) -> None:
    path = params.get("pr_curve_path", "data/08_reporting/pr_curve.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    ax.axhline(float(np.mean(y_true)), ls="--", color="grey", label="Base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (validation)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP / explicações globais (W7)
# ─────────────────────────────────────────────────────────────────────────────


def _unwrap_for_shap(model: Any) -> Any:
    """Retorna o estimador subjacente por baixo de CalibratedClassifierCV."""
    if isinstance(model, CalibratedClassifierCV):
        calibrated = getattr(model, "calibrated_classifiers_", [])
        if calibrated:
            est = getattr(calibrated[0], "estimator", None)
            if est is not None:
                return est
    return model


def _normalise_shap(shap_values: Any) -> np.ndarray:
    """Reduz output SHAP a matriz 2-D (n_samples, n_features) da classe 1."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def generate_shap_explanations(
    model: Any,
    X_test: pd.DataFrame,
    params: dict[str, Any],
    y_test: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula importância global de features com SHAP (W7).

    SHAP (SHapley Additive exPlanations) quantifica a contribuição de cada
    feature para cada previsão, fornecendo explicabilidade local e global.
    Escolha do explainer:
    * Árvores (RF/GBM/HistGB/LightGBM) -> ``shap.TreeExplainer`` (exato, rápido).
    * WOE+LogReg scorecard (Pipeline) -> ``shap.LinearExplainer`` nas features WOE.
    * Outros / falha -> permutation importance ou ``feature_importances_``.

    Gera beeswarm + bar plots e regista no MLflow (W2+W7 integração).

    Returns:
        DataFrame ``[feature, mean_abs_shap]`` ordenado descendentemente.
    """
    sample_size = min(int(params.get("shap_sample_size", 500)), len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=params.get("random_state", 42))
    estimator = _unwrap_for_shap(model)
    feature_names = list(X_sample.columns)

    summary_path = params.get("shap_summary_path", "data/08_reporting/shap_summary.png")
    bar_path = params.get("shap_importance_path", "data/08_reporting/shap_importance.png")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    is_tree = (
        isinstance(estimator, _TREE_TYPES)
        or estimator.__class__.__name__ == "LGBMClassifier"
    )

    importance_df: pd.DataFrame | None = None
    beeswarm_made = False

    try:
        if is_tree:
            explainer = shap.TreeExplainer(estimator)
            shap_matrix = _normalise_shap(explainer.shap_values(X_sample))
            plot_data = X_sample
        elif isinstance(estimator, Pipeline):
            pre, final = estimator[:-1], estimator[-1]
            X_woe = np.asarray(pre.transform(X_sample))
            explainer = shap.LinearExplainer(final, X_woe)
            shap_matrix = _normalise_shap(explainer.shap_values(X_woe))
            plot_data = pd.DataFrame(X_woe, columns=feature_names, index=X_sample.index)
        else:
            explainer = shap.LinearExplainer(estimator, X_sample)
            shap_matrix = _normalise_shap(explainer.shap_values(X_sample))
            plot_data = X_sample

        mean_abs = np.abs(shap_matrix).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        )

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_matrix, plot_data, show=False, plot_type="dot", max_display=20)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        beeswarm_made = True

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_matrix, plot_data, show=False, plot_type="bar", max_display=20)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=120, bbox_inches="tight")
        plt.close("all")

    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP falhou (%s); usando importância model-agnostic.", exc)
        importance_df = _fallback_importance(estimator, X_sample, feature_names, y_test, params)

    importance_df = (
        importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    )

    if not beeswarm_made:
        _bar_plot_from_importance(importance_df, bar_path)

    with mlflow.start_run(
        run_name="shap_explanations",
        nested=mlflow.active_run() is not None,
    ):
        for path in (summary_path, bar_path):
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="shap")

    logger.info(
        "Top 5 features: %s",
        importance_df.head(5).to_dict("records"),
    )
    return importance_df


def _fallback_importance(
    estimator: Any,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    y_test: pd.DataFrame | None,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Importância model-agnostic quando SHAP não pode ser usado."""
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    elif y_test is not None:
        y_arr = y_test.loc[X_sample.index].values.ravel()
        result = permutation_importance(
            estimator, X_sample, y_arr, n_repeats=5,
            random_state=params.get("random_state", 42), scoring="roc_auc",
        )
        values = result.importances_mean
    else:
        values = np.zeros(len(feature_names))

    if len(values) != len(feature_names):
        values = np.resize(values, len(feature_names))
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": values})


def _bar_plot_from_importance(importance_df: pd.DataFrame, bar_path: str) -> None:
    top = importance_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1B7A6E")
    ax.set_xlabel("Importance")
    ax.set_title("Global feature importance")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
