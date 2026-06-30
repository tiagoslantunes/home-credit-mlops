"""
Data Feature Engineering pipeline nodes.

Three public nodes (called by pipeline.py in order):
  1. create_features   - financial ratios, age/time, EXT_SOURCE, doc flags, encoding
                         (OHE vocabulary and target-encoding stats fitted on train only)
  2. select_features   - RFE (sklearn) feature selection on train; applied to val & test
  3. to_feature_store  - upload all three splits to Hopsworks; no-op if key absent
"""

import datetime
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# Helpers for feature construction 


def _add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    income  = df["AMT_INCOME_TOTAL"].clip(lower=1)
    credit  = df["AMT_CREDIT"].clip(lower=1)
    annuity = df["AMT_ANNUITY"].clip(lower=1)
    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]  / income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"]  / annuity
    df["INCOME_PER_PERSON"]    = income / df["CNT_FAM_MEMBERS"].clip(lower=1)
    df["PAYMENT_RATE"]         = annuity / credit
    df["CHILDREN_RATIO"]       = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].clip(lower=1)
    return df


def _add_days_features(df: pd.DataFrame) -> pd.DataFrame:
    df["AGE_YEARS"]             = -df["DAYS_BIRTH"]        / 365.25
    df["EMPLOYED_YEARS"]        = -df["DAYS_EMPLOYED"]     / 365.25
    df["REGISTRATION_YEARS"]    = -df["DAYS_REGISTRATION"] / 365.25
    df["ID_PUBLISH_YEARS"]      = -df["DAYS_ID_PUBLISH"]   / 365.25
    df["EMPLOYED_TO_AGE_RATIO"] = df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + 1e-8)
    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        df["PHONE_CHANGE_YEARS"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25
    return df


def _add_ext_source_features(df: pd.DataFrame, ext_cols: List[str]) -> pd.DataFrame:
    avail = [c for c in ext_cols if c in df.columns]
    if not avail:
        return df
    ext = df[avail]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"]  = ext.min(axis=1)
    df["EXT_SOURCE_MAX"]  = ext.max(axis=1)
    df["EXT_SOURCE_STD"]  = ext.std(axis=1).fillna(0)
    return df


def _add_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCS_COUNT"] = df[doc_cols].sum(axis=1)
    inq_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if inq_cols:
        df["BUREAU_INQUIRIES_TOTAL"] = df[inq_cols].sum(axis=1)
    return df


def _encode_binary(df: pd.DataFrame, binary_map: Dict) -> pd.DataFrame:
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def _encode_ohe(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cat_cols = train.select_dtypes(include="object").columns.tolist()
    low_card = [c for c in cat_cols if train[c].nunique() <= threshold]
    if not low_card:
        return train, val, test

    # Generate dummies from the categorical subset only - avoids carrying non-dummy
    # columns (e.g. TARGET) into the alignment and accidentally adding them to test.
    train_dummies = pd.get_dummies(train[low_card], dtype=int)
    val_dummies   = pd.get_dummies(val[[c for c in low_card if c in val.columns]], dtype=int)
    test_dummies  = pd.get_dummies(test[[c for c in low_card if c in test.columns]], dtype=int)

    # Align val and test dummies to train vocabulary; unseen categories → 0
    _, val_dummies  = train_dummies.align(val_dummies,  join="left", axis=1, fill_value=0)
    _, test_dummies = train_dummies.align(test_dummies, join="left", axis=1, fill_value=0)

    # Drop original categorical columns and append dummies
    train = pd.concat([train.drop(columns=low_card), train_dummies], axis=1)
    val   = pd.concat([val.drop(columns=low_card, errors="ignore"), val_dummies], axis=1)
    test  = pd.concat(
        [test.drop(columns=[c for c in low_card if c in test.columns]), test_dummies],
        axis=1,
    )
    return train, val, test


def _encode_target(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    target_col: str,
    smoothing: int,
    n_folds: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Smoothed cross-validated target encoding - statistics derived from train only."""
    train = train.copy()
    val   = val.copy()
    test  = test.copy()
    global_mean = train[target_col].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for col in cols:
        if col not in train.columns:
            continue

        # OOF encoding for train (no leakage within train folds)
        oof = np.full(len(train), global_mean, dtype=float)
        for tr_idx, oof_idx in kf.split(train):
            fold_tr = train.iloc[tr_idx]
            stats   = fold_tr.groupby(col)[target_col].agg(["count", "mean"])
            smooth  = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            )
            oof[oof_idx] = train.iloc[oof_idx][col].map(smooth).fillna(global_mean)

        # Full-train statistics used for val and test
        stats_all = train.groupby(col)[target_col].agg(["count", "mean"])
        full_map  = (
            (stats_all["count"] * stats_all["mean"] + smoothing * global_mean)
            / (stats_all["count"] + smoothing)
        )

        train[col + "_TE"] = oof
        val[col + "_TE"]   = val[col].map(full_map).fillna(global_mean)
        if col in test.columns:
            test[col + "_TE"] = test[col].map(full_map).fillna(global_mean)

    train = train.drop(columns=[c for c in cols if c in train.columns])
    val   = val.drop(  columns=[c for c in cols if c in val.columns])
    test  = test.drop( columns=[c for c in cols if c in test.columns])
    return train, val, test


# Public Kedro nodes 


def create_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Feature engineering on the three application-table splits.

    Encoding statistics (OHE vocabulary, target-encoding maps) are derived
    from *train only* and applied to validation and test - prevents leakage.
    The test split has no TARGET column; this is handled transparently.

    Returns
    -------
    train_features, validation_features, test_features
    """
    train = train.copy()
    val   = validation.copy()
    test  = test.copy()
    target_col = parameters["target_column"]

    train = _add_financial_ratios(train)
    val   = _add_financial_ratios(val)
    test  = _add_financial_ratios(test)

    train = _add_days_features(train)
    val   = _add_days_features(val)
    test  = _add_days_features(test)

    ext_cols = parameters.get("ext_source_cols", ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
    train = _add_ext_source_features(train, ext_cols)
    val   = _add_ext_source_features(val,   ext_cols)
    test  = _add_ext_source_features(test,  ext_cols)

    train = _add_flag_features(train)
    val   = _add_flag_features(val)
    test  = _add_flag_features(test)

    binary_map = parameters.get("binary_map", {})
    train = _encode_binary(train, binary_map)
    val   = _encode_binary(val,   binary_map)
    test  = _encode_binary(test,  binary_map)

    threshold = parameters.get("ohe_cardinality_threshold", 10)
    train, val, test = _encode_ohe(train, val, test, threshold)

    te_cols = parameters.get("target_encoding_cols", [])
    if te_cols and target_col in train.columns:
        train, val, test = _encode_target(
            train, val, test, te_cols, target_col,
            smoothing=parameters.get("target_encoding_smoothing", 10),
            n_folds=parameters.get("target_encoding_n_folds", 5),
            random_state=parameters.get("target_encoding_random_state", 42),
        )

    # Drop any residual object columns that survived encoding
    leftover = train.select_dtypes(include="object").columns.tolist()
    if leftover:
        logger.warning("Dropping %d residual object columns: %s", len(leftover), leftover)
        train = train.drop(columns=leftover)
        val   = val.drop(  columns=leftover)
        test  = test.drop( columns=[c for c in leftover if c in test.columns])

    logger.info(
        "create_features complete - train: %s | val: %s | test: %s",
        train.shape, val.shape, test.shape,
    )
    return train, val, test


def select_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Recursive Feature Elimination using RandomForest.

    RFE is fit on training data only; the selected column set is then applied
    to validation and test - no information from either influences selection.
    The test split has no TARGET column; only X_test is returned (no y_test).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, best_columns
    """
    target_col = parameters["target_column"]
    id_col     = parameters.get("id_column", "SK_ID_CURR")

    y_train = train[[target_col]]
    y_val   = validation[[target_col]] if target_col in validation.columns else pd.DataFrame()

    # Pure feature matrices for RFE (no target, no id)
    feat_drop_train = [c for c in [target_col, id_col] if c in train.columns]
    feat_drop_val   = [c for c in [target_col, id_col] if c in validation.columns]
    feat_drop_test  = [c for c in [id_col] if c in test.columns]

    X_train_feats = train.drop(columns=feat_drop_train)
    X_val_feats   = validation.drop(columns=feat_drop_val, errors="ignore")
    X_test_feats  = test.drop(columns=feat_drop_test, errors="ignore")

    n_select   = parameters.get("n_features_to_select", 50)
    step       = parameters.get("rfe_step", 5)
    est_params = parameters.get(
        "rfe_estimator_params",
        {"n_estimators": 100, "max_depth": 5, "random_state": 42, "n_jobs": -1},
    )

    n_available = X_train_feats.shape[1]
    if n_select > n_available:
        logger.warning(
            "n_features_to_select=%d exceeds available features (%d); "
            "clamping to %d.",
            n_select, n_available, n_available,
        )
        n_select = n_available

    logger.info(
        "RFE: selecting %d from %d features (step=%d) with RandomForest ...",
        n_select, n_available, step,
    )
    estimator = RandomForestClassifier(**est_params)
    rfe       = RFE(estimator, n_features_to_select=n_select, step=step)
    rfe.fit(X_train_feats, y_train.values.ravel())

    best_cols = X_train_feats.columns[rfe.get_support()].tolist()
    logger.info("RFE selected %d features", len(best_cols))

    missing_val  = set(best_cols) - set(X_val_feats.columns)
    missing_test = set(best_cols) - set(X_test_feats.columns)
    if missing_val:
        raise ValueError(f"X_val missing columns after create_features: {missing_val}")
    if missing_test:
        raise ValueError(f"X_test missing columns after create_features: {missing_test}")

    # Prepend SK_ID_CURR alongside selected features for feature store linkage
    id_cols_train = [id_col] if id_col in train.columns else []
    id_cols_val   = [id_col] if id_col in validation.columns else []
    id_cols_test  = [id_col] if id_col in test.columns else []

    return (
        pd.concat([train[id_cols_train].reset_index(drop=True),
                   X_train_feats[best_cols].reset_index(drop=True)], axis=1),
        y_train,
        pd.concat([validation[id_cols_val].reset_index(drop=True),
                   X_val_feats[best_cols].reset_index(drop=True)], axis=1),
        y_val,
        pd.concat([test[id_cols_test].reset_index(drop=True),
                   X_test_feats[best_cols].reset_index(drop=True)], axis=1),
        best_cols,
    )


def to_feature_store(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Upload all three feature splits to Hopsworks Feature Store.

    All splits are combined into one feature group with a 'split' column
    (train / validation / test). TARGET is set to -1 for the test split
    which has no ground-truth labels.

    Reads HOPSWORKS_API_KEY and HOPSWORKS_PROJECT from .env at project root.
    Returns a 'skipped' metadata dict if the key is absent - pipeline continues.

    Returns
    -------
    metadata dict (saved to 08_reporting/feature_store_metadata.json)
    """
    from dotenv import load_dotenv
    load_dotenv(override=True)

    api_key      = os.getenv("HOPSWORKS_API_KEY", "")
    project_name = os.getenv(
        "HOPSWORKS_PROJECT",
        parameters.get("hopsworks_project", "home_credit_mlops"),
    )

    if not api_key:
        logger.warning(
            "HOPSWORKS_API_KEY not set - Feature Store upload skipped. "
            "Add it to .env at the project root."
        )
        return {"status": "skipped", "reason": "HOPSWORKS_API_KEY not set"}

    import hopsworks

    host = parameters["hopsworks_host"]

    try:
        project = hopsworks.login(host=host, api_key_value=api_key, project=project_name)
        fs = project.get_feature_store()
    except Exception as exc:
        logger.error("Hopsworks login failed (host=%s, project=%s): %s", host, project_name, exc)
        return {"status": "error", "reason": f"login failed: {exc}"}

    target_col = parameters["target_column"]
    id_col     = parameters["id_column"]
    fg_name    = parameters["hopsworks_feature_group_name"]
    fg_version = parameters["hopsworks_feature_group_version"]

    # Build combined upload df with 'split' column for easy retrieval
    train_df             = X_train.copy()
    train_df[target_col] = y_train[target_col].values
    train_df["split"]    = "train"

    val_df             = X_val.copy()
    val_df[target_col] = y_val[target_col].values
    val_df["split"]    = "validation"

    test_df             = X_test.copy()
    test_df[target_col] = -1  # sentinel: no ground-truth labels for test rows
    test_df["split"]    = "test"

    upload_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    upload_df["event_time"] = datetime.datetime.now()

    try:
        fg = fs.get_or_create_feature_group(
            name=fg_name,
            version=fg_version,
            description=(
                "Home Credit Default Risk - RFE-selected engineered features. "
                "Contains train / validation / test splits (column: split). "
                "TARGET=-1 marks unlabeled test rows."
            ),
            primary_key=[id_col] if id_col in X_train.columns else [],
            event_time="event_time",
            online_enabled=False,
            time_travel_format="HUDI",
        )
    except Exception as exc:
        logger.error("Failed to get/create feature group '%s' v%d: %s", fg_name, fg_version, exc)
        return {"status": "error", "reason": f"feature group creation failed: {exc}"}

    # Clean column names for Hopsworks
    upload_df.columns = (
        upload_df.columns
        .str.lower()
        .str.replace(r'[^a-z0-9_]', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )

    batch_size = 10000
    batches = list(range(0, len(upload_df), batch_size))
    n_batches = len(batches)
    batches_uploaded = 0
    for i in batches:
        batch = upload_df.iloc[i:i + batch_size]
        is_last = i == batches[-1]
        try:
            fg.insert(batch, write_options={"wait_for_job": is_last})
        except Exception as exc:
            logger.error(
                "Batch upload failed at batch %d/%d (rows %d–%d): %s. "
                "%d batches already committed to Hopsworks.",
                batches_uploaded + 1, n_batches, i, i + len(batch) - 1, exc,
                batches_uploaded,
            )
            return {
                "status": "partial",
                "reason": f"batch {batches_uploaded + 1}/{n_batches} failed: {exc}",
                "batches_uploaded": batches_uploaded,
                "rows_uploaded": batches_uploaded * batch_size,
            }
        batches_uploaded += 1
        logger.info("Uploaded batch %d/%d: rows %d–%d", batches_uploaded, n_batches, i, i + len(batch) - 1)

    if parameters.get("enable_statistics", False):
        try:
            fg.statistics_config = {"enabled": True, "histograms": True, "correlations": True}
            fg.update_statistics_config()
            fg.compute_statistics()
            logger.info("Statistics computation triggered for feature group '%s'.", fg_name)
        except Exception as exc:
            logger.warning("Statistics computation failed (non-fatal): %s", exc)

    metadata = {
        "status": "success",
        "project": project_name,
        "feature_group": fg_name,
        "feature_group_version": fg_version,
        "n_features": X_train.shape[1],
        "n_rows_train": len(X_train),
        "n_rows_val": len(X_val),
        "n_rows_test": len(X_test),
    }

    fv_name    = parameters["hopsworks_feature_view_name"]
    fv_version = parameters["hopsworks_feature_view_version"]

    if parameters.get("enable_feature_view", False):
        try:
            fs.get_or_create_feature_view(
                name=fv_name,
                version=fv_version,
                description="Home Credit Default Risk - full feature view (all splits)",
                labels=["target"],
                query=fg.select_all(),
            )
            logger.info("Feature View '%s' v%d created/retrieved.", fv_name, fv_version)
        except Exception as exc:
            logger.warning("Feature view creation failed (non-fatal): %s", exc)
            metadata["feature_view_warning"] = str(exc)
        metadata["feature_view"] = fv_name
        metadata["feature_view_version"] = fv_version

    logger.info("Feature Store upload complete: %s", metadata)
    return metadata
