"""
Model Evaluation Module
=======================
This module calculates all required classification metrics on the held-out test set
and produces high-resolution visual charts:
- Accuracy, Precision, Recall, F1-score (binary and macro), ROC-AUC
- Confusion Matrices (Seaborn heatmaps)
- Combined ROC Curve Comparison
- Model Comparison Bar Chart
- LSTM Training History (Loss & Accuracy across epochs)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# Configure clean aesthetic for all plots
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Calculate all required evaluation metrics on the held-out test set.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
    }

    print(f"\n[Evaluation] Results for '{model_name}':")
    print(f"  - Accuracy : {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall   : {metrics['recall']:.4f}")
    print(f"  - F1 Score : {metrics['f1']:.4f} (Macro F1: {metrics['macro_f1']:.4f})")
    print(f"  - ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  - Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str,
) -> None:
    """
    Plot and save confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Helpful (0)", "Helpful (1)"],
        yticklabels=["Not Helpful (0)", "Helpful (1)"],
        cbar=False,
    )
    plt.title(f"Confusion Matrix: {model_name}", pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="semibold")
    plt.ylabel("Actual Label", fontweight="semibold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to: {save_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    baseline_probs: np.ndarray,
    lstm_probs: np.ndarray,
    baseline_auc: float,
    lstm_auc: float,
    save_path: str = "outputs/figures/roc_curve_comparison.png",
) -> None:
    """
    Plot and save combined ROC curves for both Baseline and LSTM.
    """
    fpr_base, tpr_base, _ = roc_curve(y_true, baseline_probs)
    fpr_lstm, tpr_lstm, _ = roc_curve(y_true, lstm_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_base, tpr_base, color="#2b5c8f", lw=2.2, label=f"Baseline (LR + TF-IDF) [AUC = {baseline_auc:.4f}]")
    plt.plot(fpr_lstm, tpr_lstm, color="#e65100", lw=2.2, label=f"LSTM Memory Network [AUC = {lstm_auc:.4f}]")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="semibold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="semibold")
    plt.title("ROC Curve Comparison: Baseline vs LSTM", pad=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved ROC curves to: {save_path}")


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = "outputs/figures/model_comparison_chart.png",
) -> None:
    """
    Generate side-by-side grouped bar chart comparing performance metrics.
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    melted = pd.melt(
        comparison_df,
        id_vars=["model_name"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )
    # Format labels nicely
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-Score",
        "roc_auc": "ROC-AUC",
    }
    melted["Metric"] = melted["Metric"].map(metric_labels)

    plt.figure(figsize=(9, 5.5))
    palette = ["#1976D2", "#FF9800"]
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="model_name", palette=palette)
    
    plt.title("Model Performance Comparison (Test Set)", pad=14, fontweight="bold")
    plt.xlabel("Evaluation Metric", fontweight="semibold")
    plt.ylabel("Score", fontweight="semibold")
    plt.ylim(0.0, 1.05)
    plt.legend(title="Model", frameon=True)

    # Add data labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height + 0.015),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved model comparison chart to: {save_path}")


def plot_training_history(
    history: Any,
    save_path: str = "outputs/figures/lstm_training_history.png",
) -> None:
    """
    Plot training and validation loss and accuracy across epochs.
    """
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, hist["loss"], marker="o", color="#d32f2f", label="Training Loss")
    if "val_loss" in hist:
        ax1.plot(epochs, hist["val_loss"], marker="s", color="#1976d2", label="Validation Loss")
    ax1.set_title("LSTM Loss Curves", fontweight="bold")
    ax1.set_xlabel("Epoch", fontweight="semibold")
    ax1.set_ylabel("Binary Crossentropy Loss", fontweight="semibold")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, hist["accuracy"], marker="o", color="#388e3c", label="Training Accuracy")
    if "val_accuracy" in hist:
        ax2.plot(epochs, hist["val_accuracy"], marker="s", color="#f57c00", label="Validation Accuracy")
    ax2.set_title("LSTM Accuracy Curves", fontweight="bold")
    ax2.set_xlabel("Epoch", fontweight="semibold")
    ax2.set_ylabel("Accuracy", fontweight="semibold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved LSTM training history curves to: {save_path}")


def save_evaluation_results(
    metrics_list: List[Dict[str, Any]],
    json_path: str = "outputs/evaluation/model_comparison.json",
    csv_path: str = "outputs/evaluation/model_comparison.csv",
) -> pd.DataFrame:
    """
    Save metrics list to JSON and CSV formats.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)

    df = pd.DataFrame(metrics_list)
    df.to_csv(csv_path, index=False)
    print(f"[Evaluation] Saved evaluation summaries to {json_path} and {csv_path}")
    return df


if __name__ == "__main__":
    print("Testing evaluation module...")
    y_true_toy = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    y_pred_toy = np.array([1, 1, 0, 0, 0, 1, 1, 0])
    y_prob_toy = np.array([0.9, 0.8, 0.2, 0.4, 0.1, 0.7, 0.85, 0.3])
    m = calculate_metrics(y_true_toy, y_pred_toy, y_prob_toy, "Toy Model")
    assert m["accuracy"] > 0
    print("Evaluation module test passed successfully!")
