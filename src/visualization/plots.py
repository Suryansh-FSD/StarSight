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
