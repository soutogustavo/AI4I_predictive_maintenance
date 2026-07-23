"""Training entry point."""

import joblib
import logging
from pathlib import Path

from predictive_maintenance.data import load_train_data, split_train_test
from predictive_maintenance.preprocessing import drop_non_feature_columns
from predictive_maintenance.features import add_derived_features
from predictive_maintenance.pipeline import build_pipeline
from predictive_maintenance.evaluate import evaluate_model, find_threshold_for_recall
from predictive_maintenance.visualization import (
    plot_precision_recall_curve,
    plot_precision_recall_vs_threshold,
)
from predictive_maintenance.explainability import explain_predictions
from predictive_maintenance.visualization import plot_shap_bar, plot_shap_beeswarm

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the training entry point. This is the only
    place in the package that calls basicConfig, so importing any other
    module never has side effects on logging configuration.

    Args:
        level (int): Logging level.

    Returns:
        None
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_training(
    data_path: str = "data/ai4i2020.csv",
    model_output_path: str = "models/pipeline.joblib",
    target_recall: float = 0.85,
) -> dict:
    """
    Orchestrates the full pipeline: load, preprocess, split, train,
    evaluate, and serialize.

    Args:
        data_path (str): Path to the training data.
        model_output_path (str): Path to save the trained pipeline.
        target_recall (float): Target recall level.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """

    df = load_train_data(data_path)
    df = drop_non_feature_columns(df)
    df = add_derived_features(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    threshold = find_threshold_for_recall(
        y_test,
        y_pred_proba,
        target_recall=target_recall
    )
    metrics = evaluate_model(y_test, y_pred_proba, threshold=threshold)

    plot_precision_recall_curve(y_test, y_pred_proba)
    plot_precision_recall_vs_threshold(y_test, y_pred_proba)

    shap_values, feature_names = explain_predictions(pipeline, X_test)
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]
    X_test_transformed = pipeline.named_steps["preprocessing"].transform(X_test)

    plot_shap_bar(shap_values, X_test_transformed, clean_names)
    plot_shap_beeswarm(shap_values, X_test_transformed, clean_names)

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "target_recall": target_recall,
        "feature_names": feature_names,
    }

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output_path)

    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    logger.info(f"Model artifact saved to: {model_output_path}")

    return metrics


if __name__ == "__main__":
    setup_logging()
    run_training()
