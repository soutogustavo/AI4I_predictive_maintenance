"""Module for explainability of predictions for predictive maintenance."""

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def explain_predictions(
    pipeline: Pipeline,
    X: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for a batch of raw (untransformed) input rows.
    Returns the SHAP values array and the corresponding transformed
    feature names, so each column of shap_values lines up with a name.

    Args:
        pipeline (Pipeline): Pipeline with the trained model.
        X (pd.DataFrame): DataFrame with the raw data.

    Returns:
        tuple[np.ndarray, list[str]]: Tuple containing the SHAP values array
        and the corresponding transformed feature names.
    """
    preprocessor = pipeline.named_steps["preprocessing"]
    X_transformed = preprocessor.transform(X)

    explainer = build_explainer(pipeline)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_names = get_feature_names(pipeline)
    return shap_values, feature_names


def build_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for the LightGBM model inside the pipeline.

    Args:
        pipeline (Pipeline): Pipeline with the trained model.

    Returns:
        shap.TreeExplainer: SHAP TreeExplainer for the LightGBM model.
    """
    model = pipeline.named_steps["model"]
    return shap.TreeExplainer(model)


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Recover feature names after the ColumnTransformer, since one-hot
    encoding expands 'Type' into multiple columns (Type_L, Type_M, Type_H).

    Args:
        pipeline (Pipeline): Pipeline with the trained model.

    Returns:
        list[str]: List of feature names.
    """
    preprocessor = pipeline.named_steps["preprocessing"]
    return list(preprocessor.get_feature_names_out())
