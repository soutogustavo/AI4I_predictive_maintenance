"""This module provides the inference logic for the predictive maintenance model."""

import pandas as pd
import joblib
import shap

from predictive_maintenance.features import add_derived_features


def load_artifact(model_path: str = "models/pipeline.joblib") -> dict:
    """
    Load the trained pipeline, threshold, and metadata needed for serving.

    Args:
        model_path (str): Path to the trained pipeline.

    Returns:
        dict: Dictionary containing the trained pipeline, threshold, and feature names.
    """
    return joblib.load(model_path)


def predict_risk(artifact: dict, raw_input: pd.DataFrame) -> dict:
    """
    Given raw input rows (same schema as training data, minus the target
    and leakage columns), return risk scores, flags, and SHAP-based
    explanations for each row.

    Args:
        artifact (dict): Dictionary containing the trained pipeline, threshold, and feature names.
        raw_input (pd.DataFrame): DataFrame containing the raw input data.

    Returns:
        dict: Dictionary containing the risk scores, flags, and SHAP-based explanations.
    """

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    df = add_derived_features(raw_input)
    proba = pipeline.predict_proba(df)[:, 1]
    flagged = proba >= threshold

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    X_transformed = pipeline.named_steps["preprocessing"].transform(df)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return {
        "risk_score": proba,
        "flagged": flagged,
        "shap_values": shap_values,
        "feature_names": artifact["feature_names"],
    }
