"""Nodes for the 'data_cleaning' pipeline (application_train).

The cleaning is a fit/apply pair, mirroring the train-only-statistics
philosophy of the week_04 example (encoder fit on train, reused on batch):

* ``fit_cleaning`` runs on the TRAIN split only. It learns which columns to
  drop (high-missing / highly-correlated / near-constant) and the imputation
  statistics (medians, modes), and returns them as a single ``cleaning_params``
  artifact.
* ``apply_cleaning`` is pure replay: it takes that artifact and transforms any
  frame (train, validation or a future serving batch) identically, so no
  statistic is ever computed on data the model is evaluated/served on.

The cleaning decisions themselves follow analises.ipynb (45% missing drop,
|r|>0.9 drop, near-constant drop, then the per-column imputations), with two
correct-methodology additions: the DAYS_EMPLOYED sentinel is turned into NaN +
a flag before imputation, and numeric columns are capped to the EDA value
bounds as a serving-time safeguard.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Deterministic transforms shared by fit and apply
# --------------------------------------------------------------------------- #
def _apply_days_employed_sentinel(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Replace the DAYS_EMPLOYED sentinel with NaN and add a missingness flag.

    365243 is a placeholder for "not employed" (~18% of rows). Left as a real
    value it dominates the distribution; turning it into NaN lets it be imputed
    like any other numeric column while the flag preserves the signal.
    """
    col = params["days_employed_col"]
    flag = params["days_employed_flag"]
    sentinel = params["days_employed_sentinel"]

    df = data.copy()
    if col in df.columns:
        df[flag] = (df[col] == sentinel).astype("int64")
        df[col] = df[col].replace(sentinel, np.nan)
    return df


def _protected_columns(params: dict) -> set:
    """Columns that must never be dropped by the structural-drop steps.

    These are the id/target plus every column with an explicit imputation
    decision, so the analises.ipynb imputations always have a column to act on
    (and the kept feature set is stable regardless of train-split sampling).
    """
    protected = set(params["id_columns"]) | {params["target_column"]}
    protected |= set(params.get("median_fill_cols", []))
    protected |= set(params.get("median_int_cols", []))
    protected |= set(params.get("zero_fill_cols", []))
    protected |= set(params.get("constant_fill", {}))
    protected.add(params["days_employed_flag"])
    return protected


# --------------------------------------------------------------------------- #
# Structural-drop detectors (computed on the train split)
# --------------------------------------------------------------------------- #
def _high_missing_columns(df: pd.DataFrame, threshold: float, protected: set) -> list:
    frac_missing = df.isnull().mean()
    return [c for c in frac_missing[frac_missing > threshold].index if c not in protected]


def _high_correlation_columns(df: pd.DataFrame, threshold: float, protected: set) -> list:
    """Drop the second column of each |r| > threshold numeric pair.

    Replicates the upper-triangle approach used in analises.ipynb.
    """
    numeric = [c for c in df.select_dtypes(include=["number"]).columns if c not in protected]
    if len(numeric) < 2:
        return []
    corr = df[numeric].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    redundant = {col for (_row, col), value in pairs.items() if value > threshold}
    return sorted(redundant)


def _near_constant_columns(df: pd.DataFrame, threshold: float, protected: set) -> list:
    near_constant = []
    for col in df.columns:
        if col in protected:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        if non_null.value_counts(normalize=True).iloc[0] >= threshold:
            near_constant.append(col)
    return near_constant


# --------------------------------------------------------------------------- #
# fit / apply nodes
# --------------------------------------------------------------------------- #
def fit_cleaning(train: pd.DataFrame, params: dict, numerical_rules: dict) -> dict:
    """Learn the cleaning artifact from the TRAIN split only.

    Returns a dict with the columns to drop, per-column imputation statistics
    (medians/modes) and the value-capping bounds, ready to be persisted and
    replayed by ``apply_cleaning``.
    """
    df = _apply_days_employed_sentinel(train, params)
    protected = _protected_columns(params)

    drop_missing = _high_missing_columns(df, params["missing_threshold"], protected)
    df = df.drop(columns=drop_missing)
    drop_corr = _high_correlation_columns(df, params["corr_threshold"], protected)
    df = df.drop(columns=drop_corr)
    drop_near = _near_constant_columns(df, params["near_constant_threshold"], protected)
    df = df.drop(columns=drop_near)

    cols_to_drop = drop_missing + drop_corr + drop_near

    non_feature = set(params["id_columns"]) | {params["target_column"]}
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in non_feature]
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_medians = {c: float(df[c].median()) for c in numeric_cols}
    categorical_modes = {}
    for col in categorical_cols:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            categorical_modes[col] = mode.iloc[0]

    value_bounds = {}
    if params.get("apply_value_bounds", False):
        for col, bounds in (numerical_rules or {}).items():
            if col in df.columns:
                value_bounds[col] = {"min": bounds.get("min_value"), "max": bounds.get("max_value")}

    logger.info(
        "fit_cleaning: dropped %d columns (%d high-missing, %d high-corr, %d near-constant); "
        "%d columns kept",
        len(cols_to_drop),
        len(drop_missing),
        len(drop_corr),
        len(drop_near),
        df.shape[1],
    )

    return {
        "cols_to_drop": cols_to_drop,
        "numeric_medians": numeric_medians,
        "categorical_modes": categorical_modes,
        "value_bounds": value_bounds,
    }


def apply_cleaning(data: pd.DataFrame, cleaning_params: dict, params: dict) -> pd.DataFrame:
    """Replay the learned cleaning artifact on any frame (train/validation/batch)."""
    df = _apply_days_employed_sentinel(data, params)
    df = df.drop(columns=cleaning_params["cols_to_drop"], errors="ignore")

    medians = cleaning_params["numeric_medians"]
    modes = cleaning_params["categorical_modes"]

    # Semantic zero fills (absence of record == zero events).
    for col in params.get("zero_fill_cols", []):
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Explicit new-category fills (missingness is signal).
    for col, value in params.get("constant_fill", {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # Integer-count columns -> train median rounded to nearest integer.
    for col in params.get("median_int_cols", []):
        if col in df.columns and col in medians:
            df[col] = df[col].fillna(int(round(medians[col]))).round().astype("int64")

    # Generic production fallback: numeric -> train median, categorical -> train mode.
    for col, value in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    for col, value in modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # Defensive capping to the EDA value bounds (only bites on drifted batch values).
    for col, bounds in cleaning_params.get("value_bounds", {}).items():
        if col in df.columns:
            df[col] = df[col].clip(lower=bounds.get("min"), upper=bounds.get("max"))

    # Restore int dtype on conceptually-integer columns (no NaN remain at this point).
    for col in params.get("cast_int_cols", []):
        if col in df.columns:
            df[col] = df[col].round().astype("int64")

    remaining = int(df.isnull().sum().sum())
    if remaining:
        logger.warning("apply_cleaning: %d missing values remain after imputation", remaining)
    logger.info("apply_cleaning: output shape %s", (df.shape,))
    return df
