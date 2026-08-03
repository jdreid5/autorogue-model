# evaluate.py
"""
Comprehensive evaluation for the multi-class potato leaf classifier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import config
import preprocess


def load_model(model_path: Optional[Path] = None) -> keras.Model:
    """
    Load a trained model for evaluation.
    
    Args:
        model_path: Path to model file. If None, loads the final model.
    
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        model_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME
    
    print(f"Loading model from {model_path}")
    return keras.models.load_model(model_path)


def predict_with_tta(
    model: keras.Model,
    images: np.ndarray,
    n_augments: int = config.TTA_AUGMENTS
) -> np.ndarray:
    """
    Make predictions with test-time augmentation.
    
    Args:
        model: Trained model
        images: Images to predict on
        n_augments: Number of augmented predictions to average
    
    Returns:
        Averaged predictions
    """
    # Preprocess images
    preprocessed = preprocess.preprocess_for_mobilenet(images)
    
    all_preds = []
    
    # Original prediction
    all_preds.append(model.predict(preprocessed, verbose=0))
    
    # TTA predictions
    tta_aug = preprocess.get_augmentation_layer(training=False)
    
    for _ in range(n_augments - 1):
        aug_images = tta_aug(images, training=True).numpy()
        aug_preprocessed = preprocess.preprocess_for_mobilenet(aug_images)
        all_preds.append(model.predict(aug_preprocessed, verbose=0))
    
    # Average predictions
    return np.mean(all_preds, axis=0)


def label_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 1:
        return np.argmax(labels, axis=1)
    return labels.astype(int)


def evaluate_model(
    model: keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    use_tta: bool = True,
    threshold: float = config.DECISION_THRESHOLD,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        images: Test images
        labels: True labels
        use_tta: Whether to use test-time augmentation
        threshold: Disease-vs-healthy threshold for secondary calibration
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating on {len(images)} samples...")
    
    # Get predictions
    if use_tta:
        print(f"Using Test-Time Augmentation ({config.TTA_AUGMENTS} augments)")
        y_prob = predict_with_tta(model, images)
    else:
        preprocessed = preprocess.preprocess_for_mobilenet(images)
        y_prob = model.predict(preprocessed, verbose=0)
    
    y_true = label_indices(labels)
    y_pred = np.argmax(y_prob, axis=1)
    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0,
    )

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(config.NUM_CLASSES)),
            target_names=[config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)],
            zero_division=0,
            output_dict=True,
        ),
        "per_class": {},
    }

    for idx, class_name in enumerate(config.CLASSES):
        metrics["per_class"][class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
    ).tolist()

    try:
        metrics["auc_roc_macro_ovr"] = float(
            roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["auc_roc_macro_ovr"] = None

    try:
        metrics["auc_pr_macro_ovr"] = float(
            average_precision_score(y_true_bin, y_prob, average="macro")
        )
    except ValueError:
        metrics["auc_pr_macro_ovr"] = None

    return metrics, y_prob, y_pred


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "f1"
) -> Tuple[float, Dict]:
    """
    Find optimal disease-vs-healthy threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        method: Optimization method ("f1", "youden", "precision_recall_balance")
    
    Returns:
        Optimal threshold and metrics at that threshold
    """
    thresholds = np.arange(0.1, 0.95, 0.01)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Youden's J statistic
        youden = recall + specificity - 1
        
        results.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "youden": youden,
            "pr_balance": abs(precision - recall),
        })
    
    # Find optimal based on method
    if method == "f1":
        best = max(results, key=lambda x: x["f1"])
    elif method == "youden":
        best = max(results, key=lambda x: x["youden"])
    elif method == "precision_recall_balance":
        best = min(results, key=lambda x: x["pr_balance"])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return best["threshold"], best


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)],
        yticklabels=[config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)],
        ax=ax,
        annot_kws={"size": 16}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
    for idx, class_name in enumerate(config.CLASSES):
        if y_true_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
        auc = roc_auc_score(y_true_bin[:, idx], y_prob[:, idx])
        ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
    for idx, class_name in enumerate(config.CLASSES):
        if y_true_bin[:, idx].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_prob[:, idx])
        ap = average_precision_score(y_true_bin[:, idx], y_prob[:, idx])
        ax.plot(recall, precision, lw=2, label=f"{class_name} (AP = {ap:.3f})")
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"PR curve saved to {save_path}")
    
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot metrics vs threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    thresholds = np.arange(0.1, 0.95, 0.01)
    precisions = []
    recalls = []
    f1s = []
    accuracies = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label="Precision", lw=2)
    ax.plot(thresholds, recalls, label="Recall", lw=2)
    ax.plot(thresholds, f1s, label="F1 Score", lw=2)
    ax.plot(thresholds, accuracies, label="Accuracy", lw=2, linestyle="--")
    
    # Mark optimal F1 threshold
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    ax.axvline(x=best_thresh, color="red", linestyle=":", lw=1, label=f"Best F1 @ {best_thresh:.2f}")
    
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Disease-vs-Healthy Metrics vs Threshold", fontsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Threshold analysis saved to {save_path}")
    
    return fig


def generate_report(
    metrics: Dict,
    optimal_threshold: float,
    optimal_metrics: Dict,
    save_path: Optional[Path] = None
) -> str:
    """
    Generate a text report of evaluation results.
    
    Args:
        metrics: Evaluation metrics at default threshold
        optimal_threshold: Optimal threshold found
        optimal_metrics: Metrics at optimal threshold
        save_path: Path to save the report
    
    Returns:
        Report string
    """
    report = []
    report.append("="*60)
    report.append("AUTOROGUE LEAF CLASSIFIER EVALUATION REPORT")
    report.append("="*60)
    report.append("")
    
    report.append("OVERALL METRICS")
    report.append("-"*40)
    report.append(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
    report.append(f"Macro F1:    {metrics['macro_f1']:.4f}")
    report.append(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    report.append(f"AUC-ROC OVR: {metrics['auc_roc_macro_ovr']}")
    report.append(f"AUC-PR OVR:  {metrics['auc_pr_macro_ovr']}")
    report.append("")
    
    report.append("CONFUSION MATRIX")
    report.append("-"*40)
    for row in metrics["confusion_matrix"]:
        report.append("  " + " ".join(f"{value:5d}" for value in row))
    report.append("")
    
    report.append("PER-CLASS METRICS")
    report.append("-"*40)
    for class_name, class_metrics in metrics["per_class"].items():
        report.append(
            f"{class_name}: precision={class_metrics['precision']:.4f}, "
            f"recall={class_metrics['recall']:.4f}, "
            f"f1={class_metrics['f1']:.4f}, support={class_metrics['support']}"
        )
    report.append("")
    
    report.append("DISEASE-VS-HEALTHY THRESHOLD ANALYSIS")
    report.append("-"*40)
    report.append(f"Optimal Threshold (F1): {optimal_threshold:.2f}")
    report.append(f"  Precision: {optimal_metrics['precision']:.4f}")
    report.append(f"  Recall:    {optimal_metrics['recall']:.4f}")
    report.append(f"  F1 Score:  {optimal_metrics['f1']:.4f}")
    report.append("")
    
    report.append("="*60)
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"Report saved to {save_path}")
    
    return report_str


def run_full_evaluation(
    model_path: Optional[Path] = None,
    use_tta: bool = True
) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        model_path: Path to model. If None, uses default final model.
        use_tta: Whether to use test-time augmentation
    
    Returns:
        Dictionary with all evaluation results
    """
    # Load model and data
    model = load_model(model_path)
    test_images, test_labels = preprocess.load_test_data()
    
    print(f"\nTest set: {len(test_images)} samples")
    test_label_ids = label_indices(test_labels)
    print(f"Class distribution: {dict(zip(*np.unique(test_label_ids, return_counts=True)))}")
    
    # Evaluate at default threshold
    metrics, y_prob, y_pred = evaluate_model(
        model, test_images, test_labels, use_tta=use_tta
    )
    
    # Find optimal threshold
    disease_true = (test_label_ids != config.CLASS_TO_INDEX["healthy"]).astype(int)
    disease_prob = 1.0 - y_prob[:, config.CLASS_TO_INDEX["healthy"]]
    optimal_thresh, optimal_metrics = find_optimal_threshold(disease_true, disease_prob, method="f1")
    
    # Create output directory for this evaluation
    eval_dir = config.OUTPUTS_DIR / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_confusion_matrix(
        test_label_ids, y_pred,
        save_path=eval_dir / "confusion_matrix.png"
    )
    
    plot_roc_curve(
        test_label_ids, y_prob,
        save_path=eval_dir / "roc_curve.png"
    )
    
    plot_precision_recall_curve(
        test_label_ids, y_prob,
        save_path=eval_dir / "pr_curve.png"
    )
    
    plot_threshold_analysis(
        disease_true, disease_prob,
        save_path=eval_dir / "threshold_analysis.png"
    )
    
    # Generate report
    report = generate_report(
        metrics, optimal_thresh, optimal_metrics,
        save_path=eval_dir / "evaluation_report.txt"
    )
    print("\n" + report)
    
    # Save metrics as JSON
    results = {
        "metrics": metrics,
        "optimal_threshold": optimal_thresh,
        "optimal_metrics": optimal_metrics,
        "predictions": {
            "probabilities": y_prob.tolist(),
            "predictions": y_pred.tolist(),
            "true_labels": test_label_ids.tolist(),
        }
    }
    
    with open(eval_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    plt.close("all")
    
    print(f"\nAll evaluation outputs saved to {eval_dir}")
    
    return results


if __name__ == "__main__":
    # Run full evaluation
    results = run_full_evaluation(use_tta=True)

