"""Nodes for the 'data_split' pipeline.

Stratified train/validation split of application_train, run before any
cleaning so that all imputation statistics are learned on the train partition
only (see the data_cleaning pipeline).
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(data: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split application_train into train and validation partitions.

    The split is stratified on the target by default so that the strong class
    imbalance (~11.4:1) is preserved in both partitions -- essential for a
    rare-event problem where a non-stratified split could starve the smaller
    partition of positive (default) cases.

    Args:
        data: validated application_train dataframe.
        params: ``params:data_split`` -- ``target_column``, ``test_size``,
            ``random_state`` and ``stratify``.

    Returns:
        ``(train_df, validation_df)``, both with a fresh RangeIndex.
    """
    target = params["target_column"]
    stratify = data[target] if params.get("stratify", True) else None

    train_df, val_df = train_test_split(
        data,
        test_size=params["test_size"],
        random_state=params["random_state"],
        shuffle=True,
        stratify=stratify,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_rate = train_df[target].mean()
    val_rate = val_df[target].mean()
    logger.info(
        "data_split: train=%d rows (%.4f positive), validation=%d rows (%.4f positive)",
        len(train_df),
        train_rate,
        len(val_df),
        val_rate,
    )

    return train_df, val_df
