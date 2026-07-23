"""This module handles the preprocessing of machine data for the predictive maintenance model."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLS = ["UDI", "Product ID"]


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier columns and leakage columns (failure-mode sub-labels
    that directly determine the target), keeping only the target
    'Machine failure' and genuine input features.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be processed.

    Returns:
        pd.DataFrame: DataFrame with non-feature columns dropped.
    """

    logger.info(f"Dropping {len(ID_COLS + LEAKAGE_COLS)} non-feature columns: {ID_COLS + LEAKAGE_COLS}")
    return df.drop(columns=ID_COLS + LEAKAGE_COLS)
