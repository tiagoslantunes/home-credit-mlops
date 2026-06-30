"""
Unit tests for the data_split node.

split_data performs a train split / stratified validation BEFORE cleaning,

so that no imputation statistics are learned from the validation data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit_mlops.pipelines.data_split.nodes import split_data

SPLIT_PARAMS = {
    "target_column": "TARGET",
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        "SK_ID_CURR": range(1, n + 1),
        "TARGET": ([0] * 92) + ([1] * 8),  # ~8% positivos (imbalance real ~11%)
        "FEATURE_A": np.random.normal(0, 1, n),
        "FEATURE_B": np.random.uniform(0, 1, n),
    })


class TestSplitData:
    def test_tamanhos_corretos(self, sample_df):
        """Train + validation devem somar o dataset original."""
        train, val = split_data(sample_df, SPLIT_PARAMS)
        assert len(train) + len(val) == len(sample_df)

    def test_fracao_de_validacao_aproximada(self, sample_df):
        """Validation deve ser ~20% do total."""
        _, val = split_data(sample_df, SPLIT_PARAMS)
        expected = len(sample_df) * SPLIT_PARAMS["test_size"]
        assert abs(len(val) - expected) <= 2

    def test_colunas_preservadas(self, sample_df):
        """Ambos os splits devem ter as mesmas colunas que o input."""
        train, val = split_data(sample_df, SPLIT_PARAMS)
        assert set(train.columns) == set(sample_df.columns)
        assert set(val.columns) == set(sample_df.columns)

    def test_target_presente_em_ambos_splits(self, sample_df):
        """TARGET deve estar presente em train e validation."""
        train, val = split_data(sample_df, SPLIT_PARAMS)
        assert "TARGET" in train.columns
        assert "TARGET" in val.columns

    def test_estratificacao_preserva_racio_de_classes(self, sample_df):
        """A taxa de positivos deve ser semelhante no train e no validation."""
        train, val = split_data(sample_df, SPLIT_PARAMS)
        orig_rate = sample_df["TARGET"].mean()
        assert abs(train["TARGET"].mean() - orig_rate) < 0.05
        assert abs(val["TARGET"].mean() - orig_rate) < 0.05

    def test_reproducibilidade(self, sample_df):
        """Mesmo random_state deve produzir splits idênticos."""
        train1, val1 = split_data(sample_df, SPLIT_PARAMS)
        train2, val2 = split_data(sample_df, SPLIT_PARAMS)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)

    def test_sem_sobreposicao_entre_train_e_validation(self, sample_df):
        """Nenhuma linha deve aparecer em ambos os splits."""
        train, val = split_data(sample_df, SPLIT_PARAMS)
        train_ids = set(train["SK_ID_CURR"])
        val_ids = set(val["SK_ID_CURR"])
        assert train_ids.isdisjoint(val_ids)

    def test_index_resetado(self, sample_df):
        """Ambos os splits devem ter RangeIndex começando em 0."""
        train, val = split_data(sample_df, SPLIT_PARAMS)
        assert list(train.index) == list(range(len(train)))
        assert list(val.index) == list(range(len(val)))

    def test_sem_estratificacao(self, sample_df):
        """Com stratify=False o split deve funcionar sem erro."""
        params = {**SPLIT_PARAMS, "stratify": False}
        train, val = split_data(sample_df, params)
        assert len(train) + len(val) == len(sample_df)
