"""
Testes unitários para home_credit_mlops.modeling.

Cobre o WOEEncoder (transform scorecard interpretável) e a factory
build_estimator que mantém model_selection e model_train sincronizados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from home_credit_mlops.modeling import (
    MODEL_TYPES,
    WOEEncoder,
    build_estimator,
)


@pytest.fixture
def supervised_frame():
    np.random.seed(0)
    n = 400
    X = pd.DataFrame({
        "num_signal": np.random.normal(0, 1, n),
        "num_noise": np.random.normal(0, 1, n),
        "cat": np.random.choice(["A", "B", "C"], n),
    })
    X.loc[:15, "num_signal"] = np.nan  # encoder deve tolerar missings
    y = ((X["num_signal"].fillna(0) + (X["cat"] == "A") * 1.5
          + np.random.normal(0, 0.4, n)) > 0.6).astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# WOEEncoder
# ─────────────────────────────────────────────────────────────────────────────


class TestWOEEncoder:
    def test_shape_e_sem_nan_apos_transform(self, supervised_frame):
        X, y = supervised_frame
        enc = WOEEncoder(n_bins=8).fit(X, y)
        out = enc.transform(X)
        assert out.shape == X.shape
        assert not np.isnan(out).any()

    def test_information_value_classifica_signal_acima_de_noise(self, supervised_frame):
        """A feature informativa deve ter IV superior à feature ruído."""
        X, y = supervised_frame
        enc = WOEEncoder(n_bins=8).fit(X, y)
        iv = enc.information_value().set_index("feature")["information_value"]
        assert iv["num_signal"] > iv["num_noise"]

    def test_categoria_nao_vista_mapeia_para_woe_zero(self, supervised_frame):
        """Categorias não vistas no treino devem mapear para WOE neutro (0)."""
        X, y = supervised_frame
        enc = WOEEncoder(n_bins=8).fit(X, y)
        new = X.head(5).copy()
        new["cat"] = "CATEGORIA_NOVA"
        out = pd.DataFrame(enc.transform(new), columns=X.columns)
        assert (out["cat"] == 0.0).all()

    def test_sem_leakage_dentro_de_cv(self):
        """WOEEncoder dentro de Pipeline não deve causar leakage em CV."""
        from sklearn.model_selection import cross_val_score

        X, y = make_classification(n_samples=400, n_features=8, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        pipe = build_estimator("WOELogisticRegression", {"woe__n_bins": 10})
        auc = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc").mean()
        assert auc > 0.6


# ─────────────────────────────────────────────────────────────────────────────
# build_estimator
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildEstimator:
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_todas_as_familias_treinam_e_predizem(self, model_type):
        X, y = make_classification(n_samples=200, n_features=8, random_state=1)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        est = build_estimator(model_type, {})
        est.fit(X, y)
        proba = est.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_chaves_reservadas_nao_colidem(self):
        """random_state e class_weight em best_params não devem colidir com factory."""
        est = build_estimator(
            "RandomForestClassifier",
            {"n_estimators": 10, "random_state": 7, "class_weight": "balanced"},
        )
        assert est.get_params()["random_state"] == 42  # valor da factory prevalece

    def test_model_type_desconhecido_levanta_value_error(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_estimator("ModeloInexistente", {})
