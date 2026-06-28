"""
Shared modeling utilities for Home Credit MLOps.

This module centralises model construction so that the ``model_selection`` and
``model_train`` pipelines stay in sync (DRY).  It provides:

* :class:`WOEEncoder` — a leak-free, supervised Weight-of-Evidence encoder used
  inside the interpretable WOE + Logistic Regression scorecard.  WOE/Information
  Value is the industry-standard credit-scorecard transform (Siddiqi, *Intelligent
  Credit Scoring*): it bins each feature, replaces values by the log-odds of the
  target, handles missing values and non-linearity, and yields coefficients that
  read directly as a scorecard.
* :func:`build_estimator` — a factory returning a fresh estimator for any of the
  five candidate families (RF, legacy GBM, HistGradientBoosting, LightGBM, and the
  WOE+LogReg scorecard).
* :func:`suggest_hyperparams` — the Optuna search space for each family, kept next
  to ``build_estimator`` so the two never drift apart.

The five-family comparison is motivated by Lessmann et al. (2015), *Benchmarking
state-of-the-art classification algorithms for credit scoring* (EJOR), which finds
ensemble/boosting methods consistently on top while keeping an interpretable
logistic baseline for managerial defensibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # LightGBM is a pinned dependency but keep the import soft for safety.
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]

# Canonical identifiers used across parameters_*.yml, model_selection and model_train.
MODEL_TYPES: list[str] = [
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "LGBMClassifier",
    "GradientBoostingClassifier",
    "WOELogisticRegression",
]


# ─────────────────────────────────────────────────────────────────────────────
# Weight-of-Evidence encoder
# ─────────────────────────────────────────────────────────────────────────────


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Supervised Weight-of-Evidence encoder.

    For every column the encoder builds bins (quantile bins for continuous
    features, one bin per level for low-cardinality/categorical features) and
    replaces each value with the Weight of Evidence of its bin:

        WOE(bin) = ln( P(good | bin) / P(bad | bin) )

    where *good* = ``y == 0`` and *bad* = ``y == 1``.  Laplace smoothing avoids
    division by zero.  Because ``fit`` uses ``y``, the encoder MUST be placed
    inside a scikit-learn ``Pipeline`` so cross-validation fits it on training
    folds only (no target leakage).

    Args:
        n_bins: Number of quantile bins for continuous features.
        smoothing: Laplace smoothing added to the good/bad counts per bin.
        max_cat_cardinality: Columns with at most this many distinct values are
            treated as categorical (one bin per level) rather than binned.
    """

    def __init__(
        self, n_bins: int = 10, smoothing: float = 0.5, max_cat_cardinality: int = 20
    ) -> None:
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.max_cat_cardinality = max_cat_cardinality

    def fit(self, X: Any, y: Any) -> "WOEEncoder":
        X = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y).astype(float).ravel()
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]

        total_bad = float(y_arr.sum())
        total_good = float(len(y_arr) - total_bad)

        self.numeric_edges_: dict[Any, np.ndarray] = {}
        self.woe_tables_: dict[Any, dict[str, Any]] = {}
        self.iv_: dict[Any, float] = {}

        for col in X.columns:
            s = X[col]
            is_numeric = (
                pd.api.types.is_numeric_dtype(s)
                and s.nunique(dropna=True) > self.max_cat_cardinality
            )
            if is_numeric:
                codes, inner_edges = self._fit_numeric(s)
                self.numeric_edges_[col] = inner_edges
                kind = "numeric"
            else:
                codes = self._categorical_codes(s)
                kind = "categorical"

            woe_map, iv = self._woe_for_codes(codes, y_arr, total_good, total_bad)
            self.woe_tables_[col] = {"kind": kind, "woe": woe_map}
            self.iv_[col] = iv

        return self

    def transform(self, X: Any) -> np.ndarray:
        X = pd.DataFrame(X).reset_index(drop=True)
        out = np.zeros((len(X), len(self.feature_names_in_)), dtype="float64")
        for j, col in enumerate(self.feature_names_in_):
            table = self.woe_tables_[col]
            if col in X.columns:
                s = X[col]
            else:
                s = pd.Series(np.full(len(X), np.nan), index=X.index)

            if table["kind"] == "numeric":
                codes = self._digitize(s, self.numeric_edges_[col])
            else:
                codes = self._categorical_codes(s)

            mapped = codes.map(table["woe"]).astype("float64")
            out[:, j] = mapped.fillna(0.0).to_numpy()  # unseen bin -> neutral WOE 0
        return out

    # -- internals -----------------------------------------------------------

    def _fit_numeric(self, s: pd.Series) -> tuple[pd.Series, np.ndarray]:
        s_valid = s.dropna()
        edges: np.ndarray
        try:
            _, edges = pd.qcut(
                s_valid, self.n_bins, duplicates="drop", retbins=True
            )
        except (ValueError, IndexError):
            edges = np.unique(np.quantile(s_valid, np.linspace(0, 1, self.n_bins + 1)))
        inner_edges = np.asarray(edges)[1:-1]  # drop the outer min/max edges
        return self._digitize(s, inner_edges), inner_edges

    def _digitize(self, s: pd.Series, inner_edges: np.ndarray) -> pd.Series:
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
        codes = np.digitize(arr, inner_edges, right=False).astype("float64")
        codes[np.isnan(arr)] = -1.0
        return pd.Series(codes.astype(int), index=s.index)

    @staticmethod
    def _categorical_codes(s: pd.Series) -> pd.Series:
        return s.astype("object").where(s.notna(), "__nan__").astype(str)

    def _woe_for_codes(
        self, codes: pd.Series, y_arr: np.ndarray, total_good: float, total_bad: float
    ) -> tuple[dict[Any, float], float]:
        sm = self.smoothing
        frame = pd.DataFrame({"bin": np.asarray(codes), "y": y_arr})
        woe_map: dict[Any, float] = {}
        iv = 0.0
        for bin_value, group in frame.groupby("bin"):
            bad = float(group["y"].sum())
            good = float(len(group) - bad)
            dist_good = (good + sm) / (total_good + 2 * sm)
            dist_bad = (bad + sm) / (total_bad + 2 * sm)
            woe = float(np.log(dist_good / dist_bad))
            woe_map[bin_value] = woe
            iv += (dist_good - dist_bad) * woe
        return woe_map, float(iv)

    def information_value(self) -> pd.DataFrame:
        """Return the Information Value per feature, sorted descending."""
        return (
            pd.DataFrame(
                {"feature": list(self.iv_.keys()), "information_value": list(self.iv_.values())}
            )
            .sort_values("information_value", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Estimator factory
# ─────────────────────────────────────────────────────────────────────────────


def build_estimator(
    model_type: str,
    hyperparams: dict[str, Any] | None = None,
    *,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> Any:
    """Build a fresh (unfitted) estimator for ``model_type``.

    For ``WOELogisticRegression`` a scikit-learn ``Pipeline`` is returned
    (WOE encoding -> standardisation -> logistic regression); its hyperparameters
    use the standard ``step__param`` prefixes (``woe__n_bins``, ``clf__C``).

    Args:
        model_type: One of :data:`MODEL_TYPES`.
        hyperparams: Estimator hyperparameters (already stripped of metadata).
        class_weight: Passed to families that support it (skipped for the legacy
            GradientBoostingClassifier).
        random_state: Reproducibility seed.

    Returns:
        An unfitted estimator or pipeline.

    Raises:
        ValueError: If ``model_type`` is unknown.
        ImportError: If LightGBM is requested but not installed.
    """
    _reserved = {"random_state", "class_weight", "n_jobs"}
    hp = {k: v for k, v in (hyperparams or {}).items() if k not in _reserved}

    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(
            class_weight=class_weight, random_state=random_state, n_jobs=-1, **hp
        )

    if model_type == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            class_weight=class_weight, random_state=random_state, **hp
        )

    if model_type == "LGBMClassifier":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed; cannot build LGBMClassifier.")
        lgbm_kwargs = {"subsample_freq": 1, **hp}
        return LGBMClassifier(
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **lgbm_kwargs,
        )

    if model_type == "GradientBoostingClassifier":
        # Legacy GBM has no class_weight parameter.
        return GradientBoostingClassifier(random_state=random_state, **hp)

    if model_type == "WOELogisticRegression":
        woe_kwargs = {
            k.split("__", 1)[1]: v for k, v in hp.items() if k.startswith("woe__")
        }
        clf_kwargs = {
            k.split("__", 1)[1]: v for k, v in hp.items() if k.startswith("clf__")
        }
        return Pipeline(
            [
                ("woe", WOEEncoder(**woe_kwargs)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight=class_weight,
                        random_state=random_state,
                        max_iter=1000,
                        **clf_kwargs,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unknown model_type '{model_type}'. Supported: {MODEL_TYPES}."
    )


def supports_class_weight(model_type: str) -> bool:
    """Whether ``model_type`` accepts a ``class_weight`` argument."""
    return model_type != "GradientBoostingClassifier"


# ─────────────────────────────────────────────────────────────────────────────
# Optuna search space (kept next to build_estimator so they never drift)
# ─────────────────────────────────────────────────────────────────────────────


def suggest_hyperparams(trial: Any, model_type: str) -> dict[str, Any]:
    """Return an Optuna-sampled hyperparameter dict for ``model_type``."""
    if model_type == "RandomForestClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    if model_type == "HistGradientBoostingClassifier":
        return {
            "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
        }

    if model_type == "LGBMClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        }

    if model_type == "GradientBoostingClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 160, step=40),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        }

    if model_type == "WOELogisticRegression":
        return {
            "woe__n_bins": trial.suggest_int("woe__n_bins", 5, 25),
            "clf__C": trial.suggest_float("clf__C", 1e-2, 2.0, log=True),
        }

    raise ValueError(f"Unknown model_type '{model_type}'. Supported: {MODEL_TYPES}.")
