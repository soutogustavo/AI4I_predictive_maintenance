"""Feature engineering for predictive maintenance."""

import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physically-motivated derived features. temp_diff is the
    difference between the process temperature and the air temperature,
    power is torque times rotational speed (converted to rad/s).

    Args:
        df (pd.DataFrame): DataFrame with the following columns:
            - Process temperature [K]
            - Air temperature [K]
            - Torque [Nm]
            - Rotational speed [rpm]

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            - temp_diff
            - power
    """

    df = df.copy()
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * 3.14159 / 60)

    return df
