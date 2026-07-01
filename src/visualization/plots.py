import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import logging

logger = logging.getLogger("StarSight.Plots")

def plot_diagnostic_curves(history: list, save_dir: Path) -> None:
    """
    Generate and save Loss and Accuracy curves across training/validation epochs.
    
    Args:
        history: List of epoch history records.
        save_dir: Folder path where images are written.
    """
    epochs = [r["epoch"] for r in history]
    train_losses = [r["train_loss"] for r in history]
    val_losses = [r["val_loss"] for r in history]
    train_accs = [r["train_accuracy"] for r in history]
    val_accs = [r["val_accuracy"] for r in history]
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    
    # 1. Loss curves
    axes[0].plot(epochs, train_losses, label="Train Loss", color="blue", marker="o", markersize=4)
    axes[0].plot(epochs, val_losses, label="Val Loss", color="red", linestyle="--", marker="x", markersize=4)
    axes[0].set_title("AstroNet Training vs. Validation Loss", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss (BCE)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    # 2. Accuracy curves
    axes[1].plot(epochs, train_accs, label="Train Accuracy", color="blue", marker="o", markersize=4)
    axes[1].plot(epochs, val_accs, label="Val Accuracy", color="red", linestyle="--", marker="x", markersize=4)
    axes[1].set_title("AstroNet Training vs. Validation Accuracy", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    loss_acc_path = save_dir / "loss_accuracy_curves.png"
    plt.savefig(loss_acc_path, dpi=150)
    plt.close()
    logger.info(f"Saved loss and accuracy curves to {loss_acc_path.name}")


def plot_roc_and_confusion_matrix(y_true: np.ndarray, y_pred_probs: np.ndarray, save_dir: Path) -> None:
    """
    Generate and save ROC curve and Confusion Matrix figures.
    
    Args:
        y_true: 1D NumPy ground truth label array.
        y_pred_probs: 1D predictions probability array.
        save_dir: Folder path where figures are saved.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_pred_probs >= 0.5).astype(np.int32)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    
    # 1. ROC Curve
    unique_classes = len(np.unique(y_true))
    if unique_classes < 2:
        # If only one label type is present (e.g. only 1s in small dev subset), write placeholder
        axes[0].text(0.5, 0.5, "ROC-AUC requires > 1 class label", 
                     ha="center", va="center", fontsize=11, style="italic")
        axes[0].set_title("ROC Curve (Not calculated - Single Class)", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.5)
    else:
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        axes[0].plot(fpr, tpr, color="darkorange", lw=2, label="ROC Curve")
        axes[0].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate", fontsize=10)
        axes[0].set_ylabel("True Positive Rate", fontsize=10)
        axes[0].set_title("AstroNet ROC Curve", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="lower right")
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Planet (0)", "Planet (1)"])
    disp.plot(ax=axes[1], cmap=plt.cm.Blues, values_format="d")
    axes[1].set_title("AstroNet Confusion Matrix", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    roc_cm_path = save_dir / "roc_confusion_matrix.png"
    plt.savefig(roc_cm_path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC and Confusion Matrix to {roc_cm_path.name}")


def plot_model_comparison(
    y_true: np.ndarray,
    cnn_probs: np.ndarray,
    hybrid_probs: np.ndarray,
    cnn_metrics: dict,
    hybrid_metrics: dict,
    save_dir: Path
) -> None:
    """
    Generate and save comprehensive comparison plots between the standalone CNN 
    and the hybrid CNN + LightGBM model.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard thresholding for prediction labels
    y_pred_cnn = (cnn_probs >= 0.5).astype(np.int32)
    y_pred_hybrid = (hybrid_probs >= 0.5).astype(np.int32)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    
    # 1. Core Metrics Comparison Bar Chart
    metrics_keys = ["accuracy", "precision", "recall", "f1_score"]
    cnn_vals = [cnn_metrics[k] for k in metrics_keys]
    hybrid_vals = [hybrid_metrics[k] for k in metrics_keys]
    
    x = np.arange(len(metrics_keys))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, cnn_vals, width, label="CNN Only", color="#1f77b4")
    axes[0, 0].bar(x + width/2, hybrid_vals, width, label="CNN + LightGBM", color="#2ca02c")
    axes[0, 0].set_ylabel("Score", fontsize=10)
    axes[0, 0].set_title("Classification Metrics Comparison", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([k.upper().replace("_", " ") for k in metrics_keys], fontsize=9)
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0, 0].legend()
    
    # 2. Combined ROC Curves
    unique_classes = len(np.unique(y_true))
    if unique_classes < 2:
        axes[0, 1].text(0.5, 0.5, "ROC-AUC requires > 1 class label", 
                     ha="center", va="center", fontsize=11, style="italic")
        axes[0, 1].set_title("ROC Curves Comparison (No calculation)", fontsize=12, fontweight="bold")
    else:
        fpr_cnn, tpr_cnn, _ = roc_curve(y_true, cnn_probs)
        fpr_hyb, tpr_hyb, _ = roc_curve(y_true, hybrid_probs)
        
        axes[0, 1].plot(fpr_cnn, tpr_cnn, color="#1f77b4", lw=2, 
                        label=f"CNN Only (AUC={cnn_metrics['roc_auc']:.4f})")
        axes[0, 1].plot(fpr_hyb, tpr_hyb, color="#2ca02c", lw=2, 
                        label=f"CNN + LightGBM (AUC={hybrid_metrics['roc_auc']:.4f})")
        axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel("False Positive Rate", fontsize=10)
        axes[0, 1].set_ylabel("True Positive Rate", fontsize=10)
        axes[0, 1].set_title("ROC Curves Comparison", fontsize=12, fontweight="bold")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)
        axes[0, 1].legend(loc="lower right")
        
    # 3. Execution Times Comparison
    time_keys = ["Training Time (s)", "Inference Time (s)"]
    cnn_times = [cnn_metrics.get("training_time", 0), cnn_metrics.get("inference_time", 0)]
    hybrid_times = [hybrid_metrics.get("training_time", 0), hybrid_metrics.get("inference_time", 0)]
    
    x_t = np.arange(len(time_keys))
    axes[1, 0].bar(x_t - width/2, cnn_times, width, label="CNN Only", color="#1f77b4")
    axes[1, 0].bar(x_t + width/2, hybrid_times, width, label="CNN + LightGBM", color="#2ca02c")
    axes[1, 0].set_ylabel("Seconds", fontsize=10)
    axes[1, 0].set_title("Execution Times Comparison", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(x_t)
    axes[1, 0].set_xticklabels(time_keys, fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1, 0].legend()
    
    # 4. Confusion Matrices Breakdowns
    cm_cnn = confusion_matrix(y_true, y_pred_cnn, labels=[0, 1])
    cm_hyb = confusion_matrix(y_true, y_pred_hybrid, labels=[0, 1])
    
    pred_categories = ["True Neg", "False Pos", "False Neg", "True Pos"]
    cnn_cm_flat = cm_cnn.flatten()
    hyb_cm_flat = cm_hyb.flatten()
    
    if len(cnn_cm_flat) < 4:
        cnn_cm_flat = np.zeros(4)
    if len(hyb_cm_flat) < 4:
        hyb_cm_flat = np.zeros(4)
        
    x_c = np.arange(len(pred_categories))
    axes[1, 1].bar(x_c - width/2, cnn_cm_flat, width, label="CNN Only", color="#1f77b4")
    axes[1, 1].bar(x_c + width/2, hyb_cm_flat, width, label="CNN + LightGBM", color="#2ca02c")
    axes[1, 1].set_ylabel("Sample Count", fontsize=10)
    axes[1, 1].set_title("Confusion Matrix Breakdowns", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x_c)
    axes[1, 1].set_xticklabels(pred_categories, fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1, 1].legend()
    
    plt.tight_layout()
    comp_path = save_dir / "model_comparison_plots.png"
    plt.savefig(comp_path, dpi=150)
    plt.close()
    logger.info(f"Saved comprehensive model comparison curves to {comp_path.name}")

