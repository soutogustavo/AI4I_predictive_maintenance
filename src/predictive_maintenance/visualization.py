"""This module handles the visualization for predictive maintenance."""

from pathlib import Path
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_precision_recall_curve(
    y_true,
    y_pred_proba,
    target_recall: float = 0.85,
    output_path: str = "reports/figures/precision_recall_curve.png",
    plot_fig: bool = False,
    save_fig: bool = True
) -> None:
    """
    Plot the precision-recall curve with marked operating points at
    given target recall level, against the random-baseline precision
    (the positive class rate). Saves the figure to disk.

    Args:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        target_recall (float): Target recall level.
        output_path (str): Path to save the figure.
        plot_fig (bool): Whether to plot the figure.
        save_fig (bool): Whether to save the figure.

    Returns:
        None
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, linewidth=2.2, label=f"Model (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline, linestyle="--", linewidth=1.3, label=f"Random baseline ({baseline:.3f})")

    valid = recalls[:-1] >= target_recall
    idx = np.where(valid)[0][np.argmax(thresholds[valid])]
    ax.scatter(recalls[idx], precisions[idx], color="#16a34a", s=90, zorder=5, edgecolor="white")
    ax.annotate(f"{int(target_recall*100)}% recall", (recalls[idx], precisions[idx]),
                textcoords="offset points",
                xytext=(10, 8),
                fontsize=9,
                color="#16a34a",
                fontweight="bold"
    )

    ax.set_xlabel("Recall (share of real failures detected)")
    ax.set_ylabel("Precision (share of alerts that are real failures)")
    ax.set_title("Precision-Recall Curve (Machine Failure Detection)")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if plot_fig:
        plt.show()

    if save_fig:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    plt.close(fig)


def plot_precision_recall_vs_threshold(
    y_true,
    y_pred_proba,
    target_recall: float = 0.85,
    output_path: str = "reports/figures/precision_recall_vs_threshold.png",
    plot_fig: bool = False,
    save_fig: bool = True
) -> None:
    """
    Plot precision and recall as a function of the decision threshold,
    with vertical markers at the thresholds corresponding to given
    target recall level. Saves the figure to disk.

    Args:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        target_recall (float): Target recall level.
        output_path (str): Path to save the figure.
        plot_fig (bool): Whether to plot the figure.
        save_fig (bool): Whether to save the figure.

    Returns:
        None
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(thresholds, precisions[:-1], linewidth=2.2, label="Precision")
    ax.plot(thresholds, recalls[:-1], linewidth=2.2, label="Recall")

    valid = recalls[:-1] >= target_recall
    idx = np.where(valid)[0][np.argmax(thresholds[valid])]
    ax.axvline(
        thresholds[idx], color="#16a34a", linestyle=":",
        label=f"Target recall: {int(target_recall*100)}%",
        linewidth=1.3,
        alpha=0.8
    )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall x Decision Threshold")
    ax.legend(loc="center right", fontsize=9.5)
    ax.set_xlim(0, min(1.0, thresholds.max() * 1.05))
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if plot_fig:
        plt.show()

    if save_fig:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    plt.close(fig)


def plot_shap_bar(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    output_path: str = "reports/figures/shap_bar.png",
    plot_fig: bool = False,
    save_fig: bool = True
) -> None:
    """
    Global feature importance bar plot based on mean absolute SHAP value.
    Answers "which features matter most overall" for the model.

    Args:
        shap_values (np.ndarray): SHAP values.
        X_transformed (np.ndarray): Transformed feature matrix.
        feature_names (list[str]): Feature names.
        output_path (str): Path to save the figure.
        plot_fig (bool): Whether to plot the figure.
        save_fig (bool): Whether to save the figure.

    Returns:
        None
    """
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar", show=False
    )
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()

    if plot_fig:
        plt.show()

    if save_fig:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160, bbox_inches="tight")

    plt.close(fig)


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    output_path: str = "reports/figures/shap_beeswarm.png",
    plot_fig: bool = False,
    save_fig: bool = True
) -> None:
    """
    Beeswarm plot showing both the magnitude and direction of each
    feature's effect across all test predictions, and how the feature's
    own value (color) relates to that effect. Answers "how does this
    feature drive risk up or down, and for which value ranges".

    Args:
        shap_values (np.ndarray): SHAP values.
        X_transformed (np.ndarray): Transformed feature matrix.
        feature_names (list[str]): Feature names.
        output_path (str): Path to save the figure.
        plot_fig (bool): Whether to plot the figure.
        save_fig (bool): Whether to save the figure.

    Returns:
        None
    """
    fig = plt.figure(figsize=(8, 6))

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False
    )

    plt.title("SHAP Summary (Beeswarm)")
    plt.tight_layout()

    if plot_fig:
        plt.show()

    if save_fig:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160, bbox_inches="tight")

    plt.close(fig)
