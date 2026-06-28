"""
Testes unitários para os nodes da pipeline model_train.

MLflow configurado para usar diretório temporário local.
Usa datasets toy para velocidade de execução.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import mlflow

from home_credit_mlops.pipelines.model_train.nodes import (
    evaluate_and_log_model,
    generate_shap_explanations,
    select_decision_threshold,
    train_final_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_PARAMS = {
    "random_state": 42,
    "class_weight": "balanced",
    "shap_sample_size": 20,
    "shap_summary_path": "data/08_reporting/shap_summary_test.png",
    "shap_importance_path": "data/08_reporting/shap_importance_test.png",
    "confusion_matrix_path": "data/08_reporting/confusion_matrix_test.png",
    "calibration_curve_path": "data/08_reporting/calibration_curve_test.png",
    "pr_curve_path": "data/08_reporting/pr_curve_test.png",
    "mlflow_registered_model_name": "test_home_credit_model",
    "cost_false_negative": 10.0,
    "cost_false_positive": 1.0,
}

RF_BEST_PARAMS = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 10,
    "max_depth": 3,
}


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test_model_train")
    yield
    mlflow.end_run()


@pytest.fixture
def toy_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y_df = pd.DataFrame({"TARGET": y})
    X_train, X_test = X_df.iloc[:160], X_df.iloc[160:]
    y_train, y_test = y_df.iloc[:160], y_df.iloc[160:]
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# train_final_model
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainFinalModel:
    def test_modelo_treinado_tem_predict_proba(self, toy_data):
        X_train, _, y_train, _ = toy_data
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, TRAIN_PARAMS)
        assert hasattr(model, "predict_proba")

    def test_tipo_de_modelo_respeitado(self, toy_data):
        from sklearn.ensemble import RandomForestClassifier
        X_train, _, y_train, _ = toy_data
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, TRAIN_PARAMS)
        assert isinstance(model, RandomForestClassifier)

    def test_model_type_em_falta_levanta_key_error(self, toy_data):
        X_train, _, y_train, _ = toy_data
        with pytest.raises(KeyError, match="model_type"):
            train_final_model(X_train, y_train, {"n_estimators": 10}, TRAIN_PARAMS)

    def test_metadata_da_selecao_ignorada_no_treino(self, toy_data):
        """cv_score e search_strategy não devem ser passados ao sklearn."""
        X_train, _, y_train, _ = toy_data
        params_com_metadata = {
            **RF_BEST_PARAMS,
            "cv_score": 0.74,
            "search_strategy": "GridSearchCV",
        }
        model = train_final_model(X_train, y_train, params_com_metadata, TRAIN_PARAMS)
        assert hasattr(model, "predict_proba")

    def test_calibracao_envolve_modelo_em_calibrated_cv(self, toy_data):
        from sklearn.calibration import CalibratedClassifierCV
        X_train, _, y_train, _ = toy_data
        params = {**TRAIN_PARAMS, "calibrate": True, "calibration_cv": 3}
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, params)
        assert isinstance(model, CalibratedClassifierCV)

    def test_woe_logistic_regression_treina(self, toy_data):
        X_train, _, y_train, _ = toy_data
        model = train_final_model(
            X_train, y_train, {"model_type": "WOELogisticRegression"}, TRAIN_PARAMS
        )
        assert hasattr(model, "predict_proba")


# ─────────────────────────────────────────────────────────────────────────────
# select_decision_threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestSelectDecisionThreshold:
    def test_devolve_threshold_entre_0_e_1(self, toy_data):
        X_train, _, y_train, _ = toy_data
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, TRAIN_PARAMS)
        artifact = select_decision_threshold(model, X_train, y_train, TRAIN_PARAMS)

        assert 0.0 <= artifact["threshold"] <= 1.0
        assert "f1_positive" in artifact

    def test_estrategia_custo_minimiza_falsos_negativos(self, toy_data):
        """Estratégia 'cost' deve retornar threshold e custo esperado."""
        X_train, _, y_train, _ = toy_data
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, TRAIN_PARAMS)
        artifact = select_decision_threshold(
            model, X_train, y_train,
            {**TRAIN_PARAMS, "threshold_strategy": "cost"}
        )
        assert artifact["strategy"] == "cost"
        assert "expected_cost" in artifact
        assert 0.0 <= artifact["threshold"] <= 1.0

    def test_estrategia_f1_macro(self, toy_data):
        X_train, _, y_train, _ = toy_data
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, TRAIN_PARAMS)
        artifact = select_decision_threshold(
            model, X_train, y_train,
            {**TRAIN_PARAMS, "threshold_strategy": "f1_macro"}
        )
        assert artifact["strategy"] == "f1_macro"


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_and_log_model
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateAndLogModel:
    def test_roc_auc_acima_de_baseline(self, toy_data, tmp_path):
        X_train, X_test, y_train, y_test = toy_data
        params = {**TRAIN_PARAMS, "confusion_matrix_path": str(tmp_path / "cm.png"),
                  "calibration_curve_path": str(tmp_path / "cal.png"),
                  "pr_curve_path": str(tmp_path / "pr.png")}
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, params)
        metrics = evaluate_and_log_model(model, X_train, y_train, X_test, y_test, params)

        assert metrics["test_roc_auc"] > 0.6

    def test_metricas_credit_scoring_presentes(self, toy_data, tmp_path):
        """Painel de métricas deve incluir Gini, KS, PR-AUC, Brier."""
        X_train, X_test, y_train, y_test = toy_data
        params = {**TRAIN_PARAMS, "confusion_matrix_path": str(tmp_path / "cm.png"),
                  "calibration_curve_path": str(tmp_path / "cal.png"),
                  "pr_curve_path": str(tmp_path / "pr.png")}
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, params)
        metrics = evaluate_and_log_model(model, X_train, y_train, X_test, y_test, params)

        for key in ("test_gini", "test_ks", "test_pr_auc", "test_brier",
                    "test_log_loss", "test_expected_cost"):
            assert key in metrics

    def test_gini_e_transformacao_monotona_de_auc(self, toy_data, tmp_path):
        X_train, X_test, y_train, y_test = toy_data
        params = {**TRAIN_PARAMS, "confusion_matrix_path": str(tmp_path / "cm.png"),
                  "calibration_curve_path": str(tmp_path / "cal.png"),
                  "pr_curve_path": str(tmp_path / "pr.png")}
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, params)
        metrics = evaluate_and_log_model(model, X_train, y_train, X_test, y_test, params)

        assert abs(metrics["test_gini"] - (2 * metrics["test_roc_auc"] - 1)) < 1e-9

    def test_threshold_de_producao_usado_nas_metricas(self, toy_data, tmp_path):
        """Métricas threshold-dependentes devem usar o threshold selecionado."""
        X_train, X_test, y_train, y_test = toy_data
        params = {**TRAIN_PARAMS, "confusion_matrix_path": str(tmp_path / "cm.png"),
                  "calibration_curve_path": str(tmp_path / "cal.png"),
                  "pr_curve_path": str(tmp_path / "pr.png")}
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, params)
        threshold = select_decision_threshold(model, X_train, y_train, params)
        metrics = evaluate_and_log_model(
            model, X_train, y_train, X_test, y_test, params, threshold
        )

        assert metrics["decision_threshold"] == threshold["threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# generate_shap_explanations
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateShapExplanations:
    def test_shape_do_dataframe_de_importancia(self, toy_data, tmp_path):
        """Deve haver uma linha por feature no DataFrame de importâncias."""
        X_train, X_test, y_train, _ = toy_data
        params = {**TRAIN_PARAMS,
                  "shap_summary_path": str(tmp_path / "beeswarm.png"),
                  "shap_importance_path": str(tmp_path / "bar.png")}
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, params)
        importance_df = generate_shap_explanations(model, X_test, params)

        assert len(importance_df) == X_test.shape[1]
        assert set(importance_df.columns) == {"feature", "mean_abs_shap"}

    def test_importancias_ordenadas_descendentemente(self, toy_data, tmp_path):
        X_train, X_test, y_train, _ = toy_data
        params = {**TRAIN_PARAMS,
                  "shap_summary_path": str(tmp_path / "beeswarm.png"),
                  "shap_importance_path": str(tmp_path / "bar.png")}
        model = train_final_model(X_train, y_train, RF_BEST_PARAMS, params)
        importance_df = generate_shap_explanations(model, X_test, params)

        values = importance_df["mean_abs_shap"].values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_shap_funciona_para_woe_scorecard(self, toy_data, tmp_path):
        """SHAP com LinearExplainer deve funcionar para o scorecard WOE+LogReg."""
        X_train, X_test, y_train, _ = toy_data
        params = {**TRAIN_PARAMS, "shap_sample_size": 30,
                  "shap_summary_path": str(tmp_path / "s.png"),
                  "shap_importance_path": str(tmp_path / "b.png")}
        model = train_final_model(
            X_train, y_train, {"model_type": "WOELogisticRegression"}, params
        )
        importance_df = generate_shap_explanations(model, X_test, params)
        assert len(importance_df) == X_test.shape[1]
