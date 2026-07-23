"""This module contains thepipeline for predictive maintenance."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb


CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "temp_diff", "power"
]


def build_pipeline(
    class_weight: str = "balanced",
    objective: str = "binary",
    random_state: int = 42
) -> Pipeline:
    """Build the predictive maintenance pipeline.

    Args:
        class_weight (str): Weight for the positive class.
        objective (str): Objective function for the model.
        random_state (int): Random state for reproducibility.

    Returns:
        Pipeline: The predictive maintenance pipeline.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])

    model = lgb.LGBMClassifier(
        objective=objective,
        class_weight=class_weight,
        random_state=random_state
    )

    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ])
