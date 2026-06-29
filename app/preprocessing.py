"""Convert a raw application JSON payload into the model's 51-feature vector.

The training pipeline persists three artefacts the API has to replay:

  * ``cleaning_params`` (pickle) — drop list, imputation medians/modes, value bounds.
    Used directly via ``apply_cleaning`` from the data_cleaning pipeline.
  * ``best_columns`` (pickle) and ``production_model.feature_names_in_`` — the
    exact ordered feature vector the model expects.

Stateless feature transforms (financial ratios, age in years, EXT_SOURCE
aggregates, document flags, binary encoding) are imported directly from the
data_feat_engineering pipeline; they are pure functions of the input row.

Encoder-based features that DO have fit-time state:
  * One-Hot Encoded categoricals — derived deterministically from the category
    string. The column vocabulary is read once at startup from ``best_columns``,
    so unseen categories collapse to all-zeros (the same behaviour
    ``OneHotEncoder(handle_unknown='ignore')`` exhibits at training time).
  * Target Encoded categoricals (OCCUPATION_TYPE, ORGANIZATION_TYPE) — without
    the fitted mean maps, the API falls back to the global positive rate
    (~0.08). The fallback feature name is reported in the response so the
    caller knows the prediction was made with default values.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from home_credit_mlops.pipelines.data_cleaning.nodes import apply_cleaning
from home_credit_mlops.pipelines.data_feat_engineering.nodes import (
    _add_days_features,
    _add_ext_source_features,
    _add_financial_ratios,
    _add_flag_features,
    _encode_binary,
)

logger = logging.getLogger(__name__)

# Global default for target-encoded columns when the API doesn't have the
# fitted maps. ~0.0807 is the dataset positive rate on the training split.
TARGET_ENCODING_FALLBACK = 0.0807

# OHE column → (source raw column, exact category value to match).
# Derived from the model.feature_names_in_ values; this mapping is the only
# place where the OHE rules live so a change in feat_eng is a one-line edit.
_OHE_RULES: Dict[str, Tuple[str, str]] = {
    "NAME_CONTRACT_TYPE_Cash_loans": ("NAME_CONTRACT_TYPE", "Cash loans"),
    "NAME_CONTRACT_TYPE_Revolving_loans": ("NAME_CONTRACT_TYPE", "Revolving loans"),
    "CODE_GENDER_F": ("CODE_GENDER", "F"),
    "CODE_GENDER_M": ("CODE_GENDER", "M"),
    "NAME_INCOME_TYPE_Pensioner": ("NAME_INCOME_TYPE", "Pensioner"),
    "NAME_INCOME_TYPE_Working": ("NAME_INCOME_TYPE", "Working"),
    "NAME_EDUCATION_TYPE_Higher_education": ("NAME_EDUCATION_TYPE", "Higher education"),
    "NAME_EDUCATION_TYPE_Secondary_/_secondary_special": (
        "NAME_EDUCATION_TYPE",
        "Secondary / secondary special",
    ),
    "NAME_FAMILY_STATUS_Married": ("NAME_FAMILY_STATUS", "Married"),
}


def raw_to_features(
    raw: Dict[str, Any],
    *,
    cleaning_params: Dict[str, Any],
    cleaning_node_params: Dict[str, Any],
    feat_node_params: Dict[str, Any],
    model_columns: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Engineer features for a single raw application payload.

    Returns the 1-row feature matrix and a list of feature names that fell back
    to defaults (empty when every value came from the payload).
    """
    df = pd.DataFrame([raw])

    # 1. Replay the persisted cleaning artefact (handles DAYS_EMPLOYED sentinel,
    #    drops the same columns as training, imputes nulls).
    df = apply_cleaning(df, cleaning_params, cleaning_node_params)

    # 2. Stateless engineered features — identical to training.
    df = _add_financial_ratios(df)
    df = _add_days_features(df)
    ext_cols = feat_node_params.get("ext_source_cols", ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
    df = _add_ext_source_features(df, ext_cols)
    df = _add_flag_features(df)

    # 3. Binary encoding (FLAG_OWN_CAR Y/N → 1/0, etc.).
    df = _encode_binary(df, feat_node_params.get("binary_map", {}))

    # 4. OHE — deterministic from the raw category string.
    for ohe_col, (source_col, expected_value) in _OHE_RULES.items():
        raw_value = raw.get(source_col)
        df[ohe_col] = int(raw_value == expected_value) if raw_value is not None else 0

    # 5. Target-encoded columns — fall back to the dataset prior when unavailable.
    fallback: List[str] = []
    for te_col in ("OCCUPATION_TYPE_TE", "ORGANIZATION_TYPE_TE"):
        df[te_col] = TARGET_ENCODING_FALLBACK
        fallback.append(te_col)

    # 6. Ensure every model column exists, in order. Missing engineered columns
    #    become NaN here, but the model was trained without NaN, so we coerce to
    #    0.0 and flag them — this is what an OHE encoder with handle_unknown
    #    would do at training time anyway.
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0.0
            fallback.append(col)

    df = df[model_columns]
    return df, fallback
