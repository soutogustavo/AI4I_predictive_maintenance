"""This module handles the machine data for the predictive maintenance model."""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_train_data(file_path: str) -> pd.DataFrame:
    """Load training data from CSV file.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Training data.

    """

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def split_train_test(
    df: pd.DataFrame,
    target_col: str = "Machine failure",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into train/test sets, stratified by the target.

    Args:
        df (pd.DataFrame): DataFrame to split.
        target_col (str): Name of the target column.
        test_size (float): Size of the test set.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple containing
        the training and testing sets, and the training and testing targets.
    """
    logger.info(f"Splitting data: test_size={test_size}, stratify=True")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test
