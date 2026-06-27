import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import logging

logger = logging.getLogger("StarSight.Metrics")

def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, config) -> dict:
    """
    Compute core classification metrics and write results to evaluation_metrics.csv.
    Handles edge cases with single-class datasets (e.g. development subset) safely.
    
    Args:
        y_true: 1D NumPy array of ground truth labels.
        y_pred_probs: 1D NumPy array of predictions probabilities.
        config: Configuration object.
        
    Returns:
        dict: Summary of metrics.
    """
    # Convert predictions to binary labels
    y_pred = (y_pred_probs >= 0.5).astype(np.int32)
    
    # Base computations
    acc = accuracy_score(y_true, y_pred)
    
    # Handle single-class scenarios safely (which is common in development splits)
    unique_classes = len(np.unique(y_true))
    
    if unique_classes < 2:
        logger.warning("Only one binary class is present in true labels. Adjusting metrics metrics.")
        prec = 1.0 if np.sum(y_true == y_pred) == len(y_true) else 0.0
        rec = 1.0 if np.sum(y_true == y_pred) == len(y_true) else 0.0
        f1 = 1.0 if np.sum(y_true == y_pred) == len(y_true) else 0.0
        auc = 1.0  # Cannot compute ROC-AUC with single class, default to 1.0 or nan
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_pred_probs)
        except Exception as e:
            logger.warning(f"Failed to calculate ROC-AUC score: {str(e)}")
            auc = 0.5

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }
    
    # Export metrics summary to CSV
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df_metrics = pd.DataFrame([metrics])
    csv_path = config.METRICS_DIR / "evaluation_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    logger.info(f"Saved evaluation metrics summary to {csv_path.name}")
    
    return metrics
