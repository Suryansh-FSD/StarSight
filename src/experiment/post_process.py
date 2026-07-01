import matplotlib
matplotlib.use('Agg')  # Disable interactive backend for headless execution stability

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time

# Resolve base directory and setup path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root.resolve()))

import src.config as config
from src.visualization.plots import plot_diagnostic_curves, plot_roc_and_confusion_matrix, plot_model_comparison
from src.models.explainability import generate_shap_explanations
from src.models.metrics import calculate_metrics
import lightgbm as lgb

def main():
    print("[PostProcess] Starting post-experiment plotting and explainability pipeline...")
    
    # Resolve file paths from centralized config
    history_csv = config.METRICS_DIR / "history.csv"
    predictions_npz = config.PREDICTIONS_DIR / "test_predictions.npz"
    predictions_hybrid_npz = config.PREDICTIONS_DIR / "test_predictions_hybrid.npz"
    embeddings_npz = config.PREDICTIONS_DIR / "feature_embeddings.npz"
    lightgbm_txt = config.MODELS_DIR / "lightgbm_model.txt"
    
    # 1. Load History and plot diagnostic curves
    if history_csv.exists():
        print(f"[PostProcess] Loading training history from {history_csv.name}...")
        df_history = pd.read_csv(history_csv)
        history = df_history.to_dict(orient="records")
        plot_diagnostic_curves(history, config.FIGURES_DIR)
    
    # 2. Load Predictions and plot ROC/Confusion Matrix
    if predictions_npz.exists():
        print(f"[PostProcess] Loading CNN test predictions from {predictions_npz.name}...")
        cnn_data = np.load(predictions_npz)
        y_true_cnn = cnn_data["y_true"]
        y_pred_probs_cnn = cnn_data["y_pred_probs"]
        plot_roc_and_confusion_matrix(y_true_cnn, y_pred_probs_cnn, config.FIGURES_DIR)
        
    # 3. Load hybrid predictions and feature embeddings to plot comparisons
    if predictions_npz.exists() and predictions_hybrid_npz.exists():
        print("[PostProcess] Generating model comparisons...")
        cnn_data = np.load(predictions_npz)
        y_true = cnn_data["y_true"]
        cnn_probs = cnn_data["y_pred_probs"]
        
        hybrid_data = np.load(predictions_hybrid_npz)
        hybrid_probs = hybrid_data["y_pred_probs"]
        
        # Load pre-saved metrics or compute directly
        comparison_csv_path = config.METRICS_DIR / "model_comparison_metrics.csv"
        if comparison_csv_path.exists():
            df_comp = pd.read_csv(comparison_csv_path)
            # Convert comparison to dictionary metrics
            cnn_metrics = {
                "accuracy": float(df_comp.loc[df_comp["Metric"] == "ACCURACY", "CNN Only"].values[0]),
                "precision": float(df_comp.loc[df_comp["Metric"] == "PRECISION", "CNN Only"].values[0]),
                "recall": float(df_comp.loc[df_comp["Metric"] == "RECALL", "CNN Only"].values[0]),
                "f1_score": float(df_comp.loc[df_comp["Metric"] == "F1 SCORE", "CNN Only"].values[0]),
                "roc_auc": float(df_comp.loc[df_comp["Metric"] == "ROC AUC", "CNN Only"].values[0]),
                "training_time": float(df_comp.loc[df_comp["Metric"] == "TRAINING TIME", "CNN Only"].values[0]),
                "inference_time": float(df_comp.loc[df_comp["Metric"] == "INFERENCE TIME", "CNN Only"].values[0])
            }
            hybrid_metrics = {
                "accuracy": float(df_comp.loc[df_comp["Metric"] == "ACCURACY", "CNN + LightGBM"].values[0]),
                "precision": float(df_comp.loc[df_comp["Metric"] == "PRECISION", "CNN + LightGBM"].values[0]),
                "recall": float(df_comp.loc[df_comp["Metric"] == "RECALL", "CNN + LightGBM"].values[0]),
                "f1_score": float(df_comp.loc[df_comp["Metric"] == "F1 SCORE", "CNN + LightGBM"].values[0]),
                "roc_auc": float(df_comp.loc[df_comp["Metric"] == "ROC AUC", "CNN + LightGBM"].values[0]),
                "training_time": float(df_comp.loc[df_comp["Metric"] == "TRAINING TIME", "CNN + LightGBM"].values[0]),
                "inference_time": float(df_comp.loc[df_comp["Metric"] == "INFERENCE TIME", "CNN + LightGBM"].values[0])
            }
        else:
            cnn_metrics = calculate_metrics(y_true, cnn_probs)
            hybrid_metrics = calculate_metrics(y_true, hybrid_probs)
            
        plot_model_comparison(y_true, cnn_probs, hybrid_probs, cnn_metrics, hybrid_metrics, config.FIGURES_DIR)
        
    # 4. Load LightGBM model and generate SHAP explanations
    if lightgbm_txt.exists() and embeddings_npz.exists():
        print(f"[PostProcess] Loading LightGBM booster from {lightgbm_txt.name}...")
        lgb_model = lgb.Booster(model_file=str(lightgbm_txt))
        
        print(f"[PostProcess] Loading feature embeddings from {embeddings_npz.name}...")
        emb_data = np.load(embeddings_npz)
        X_train = emb_data["X_train_emb"]
        X_test = emb_data["X_test_emb"]
        
        print("[PostProcess] Generating SHAP explainability visualizations...")
        generate_shap_explanations(lgb_model, X_train, X_test, config.EXPLAINABILITY_DIR)
        
    print("[PostProcess] Post-experiment processing completed successfully!")

if __name__ == "__main__":
    main()
