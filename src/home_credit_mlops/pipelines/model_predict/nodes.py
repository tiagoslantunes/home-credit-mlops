"""Nodes for the 'model_predict' pipeline.

Two responsibilities, factored so the FastAPI app reuses the same scoring
primitive as the batch Kedro pipeline:

  - ``score_features``      — pure function (model, columns, threshold, X) → predictions DataFrame.
                              Public utility imported by ``app/main.py`` for the /predict route.
  - ``predict``             — Kedro node: thin wrapper that injects the catalog-loaded
                              model + production_columns + decision_threshold and returns
                              the batch predictions.
  - ``evaluate_predictions`` — when ground-truth labels are available (validation/test
                              only — never in production), report ROC-AUC / PR-AUC / accuracy
                              at the production threshold. Keeps the scorecard honest by
                              measuring at serving time, not just at training time.

The model is registered to MLflow via ``kedro_mlflow.io.artifacts.MlflowArtifactDataset``
in the catalog, so the pickle on disk is also the artefact tracked in the registry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public scoring primitive — reused by FastAPI                                 #
# --------------------------------------------------------------------------- #
def get_production_columns(model: Any) -> List[str]:
    """Read the exact feature list the model was trained on.

    Reads ``model.feature_names_in_`` (set by scikit-learn estimators), so the
    serving column order is always derived FROM the model — never re-declared
    elsewhere — and stays in lockstep with retraining.
    """
    if not hasattr(model, "feature_names_in_"):
        raise AttributeError(
            "Model does not expose feature_names_in_; was it fit on a DataFrame?"
        )
    return list(model.feature_names_in_)


def score_features(
    features: pd.DataFrame,
    model: Any,
    decision_threshold: float,
    id_column: str | None = None,
    production_columns: List[str] | None = None,
) -> pd.DataFrame:
    """Score a feature batch with the production model.

    Args:
        features: One row per applicant, already cleaned + feature-engineered
            (i.e. the same transformations the model was trained on). Extra
            columns are dropped; missing ones raise.
        model: Fitted scikit-learn-like classifier exposing ``predict_proba``.
        decision_threshold: Probability cutoff for the positive class.
        id_column: Optional identifier column to copy through to the output.
        production_columns: Optional explicit column order; defaults to the
            list embedded in the model (``feature_names_in_``).

    Returns:
        DataFrame with one row per input row:
          ``[id_column], probability, predicted_class``
    """
    if production_columns is None:
        production_columns = get_production_columns(model)

    # LightGBM sanitises feature names by replacing whitespace with underscores
    # at fit time, so the names recorded in ``feature_names_in_`` differ from
    # the raw CSV column names. Apply the same sanitisation here so a fresh
    # batch from disk lines up with what the model expects.
    sanitised_features = features.rename(columns=lambda c: c.replace(" ", "_"))

    missing = [c for c in production_columns if c not in sanitised_features.columns]
    if missing:
        raise ValueError(
            f"score_features: {len(missing)} required column(s) missing from input: {missing[:5]}"
        )

    # Reorder strictly — sklearn estimators rely on positional column order.
    X = sanitised_features[production_columns]

    proba = model.predict_proba(X)[:, 1]
    pred_class = (proba >= decision_threshold).astype(int)

    output = pd.DataFrame({
        "probability": proba,
        "predicted_class": pred_class,
    })
    if id_column and id_column in features.columns:
        output.insert(0, id_column, features[id_column].to_numpy())
    return output


# --------------------------------------------------------------------------- #
# Kedro node — batch scoring                                                   #
# --------------------------------------------------------------------------- #
def predict(
    features: pd.DataFrame,
    model: Any,
    decision_threshold: Dict[str, Any] | float,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Batch scoring node — thin wrapper around ``score_features``.

    ``decision_threshold`` is the JSON artefact written by ``model_train``;
    we accept either the raw float or the dict shape so we don't have to know
    the persisted format. The feature column order is read directly from
    ``model.feature_names_in_`` so no external "production_columns" artefact
    needs to stay in sync.
    """
    threshold = decision_threshold["threshold"] if isinstance(decision_threshold, dict) else float(decision_threshold)
    id_col = params.get("id_column", "SK_ID_CURR")

    preds = score_features(
        features=features,
        model=model,
        decision_threshold=threshold,
        id_column=id_col,
    )
    logger.info(
        "Batch scoring: %d rows, %.1f%% predicted positive at threshold=%.3f",
        len(preds),
        100 * preds["predicted_class"].mean(),
        threshold,
    )
    return preds


# --------------------------------------------------------------------------- #
# Optional node — serving-time evaluation when labels are available            #
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    predictions: pd.DataFrame,
    y_true: pd.DataFrame,
    decision_threshold: Dict[str, Any] | float,
) -> Dict[str, float]:
    """Compute classification metrics on the predicted batch.

    Used in batch evaluation (val/test) — NOT in production, where labels
    are not yet observed at scoring time. The same metrics the report cites
    from ``model_train`` are recomputed here so the relatório can show that
    the model behaves at serving time exactly as it did at training time.
    """
    threshold = decision_threshold["threshold"] if isinstance(decision_threshold, dict) else float(decision_threshold)

    # y_true comes through as a single-column DataFrame
    y = y_true.iloc[:, 0].to_numpy() if isinstance(y_true, pd.DataFrame) else np.asarray(y_true)
    proba = predictions["probability"].to_numpy()
    pred = predictions["predicted_class"].to_numpy()

    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "accuracy_at_threshold": float(accuracy_score(y, pred)),
        "precision_at_threshold": float(precision_score(y, pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y, pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate_predicted": float(pred.mean()),
        "positive_rate_actual": float(y.mean()),
        "n_rows": int(len(y)),
    }
    logger.info(
        "Serving-time evaluation: ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f at threshold=%.3f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1_at_threshold"],
        threshold,
    )
    return metrics
