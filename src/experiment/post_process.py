import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
    lightgbm_pkl = config.MODELS_DIR / "lightgbm.pkl"
    if (lightgbm_pkl.exists() or lightgbm_txt.exists()) and embeddings_npz.exists():
        if lightgbm_pkl.exists():
            print(f"[PostProcess] Loading LightGBMClassifier wrapper from {lightgbm_pkl.name}...")
            from src.models.lightgbm_classifier import LightGBMClassifier
            lgb_clf = LightGBMClassifier()
            lgb_clf.load(lightgbm_pkl)
            lgb_model = lgb_clf.model
        else:
            print(f"[PostProcess] Loading LightGBM booster from {lightgbm_txt.name}...")
            lgb_model = lgb.Booster(model_file=str(lightgbm_txt))
        
        print(f"[PostProcess] Loading feature embeddings from {embeddings_npz.name}...")
        emb_data = np.load(embeddings_npz)
        X_train = emb_data["X_train_emb"]
        X_test = emb_data["X_test_emb"]
        
        print("[PostProcess] Generating SHAP explainability visualizations...")
        generate_shap_explanations(lgb_model, X_train, X_test, config.EXPLAINABILITY_DIR)
        
    # 5. Load CNN model and generate Grad-CAM explanations
    if getattr(config, 'GRADCAM_ENABLED', True):
        print("[PostProcess] Preparing Grad-CAM explainability pipeline...")
        best_model_pt = config.MODELS_DIR / "best_model.pt"
        data_path = config.PROCESSED_DIR / "final_dataset.npz"
        
        if best_model_pt.exists() and data_path.exists() and predictions_npz.exists():
            import torch
            from src.models.registry import get_model
            from src.models.gradcam import generate_gradcam, save_gradcam_figures
            
            print(f"[PostProcess] Recreating train/val/test data splits from {data_path.name}...")
            # Load dataset
            data = np.load(data_path)
            X_global = data['X_global']
            X_local = data['X_local']
            X_stellar = data['X_stellar']
            y = data['y']
            
            if config.DEV_MODE:
                X_global = X_global[:4]
                X_local = X_local[:4]
                X_stellar = X_stellar[:4]
                y = y[:4]
                
            num_samples = len(y)
            indices = np.arange(num_samples)
            
            if num_samples < 5:
                train_idx = indices[:max(1, int(0.5 * num_samples))]
                val_idx = indices[len(train_idx):max(len(train_idx)+1, int(0.75 * num_samples))]
                test_idx = indices[len(train_idx)+len(val_idx):]
            else:
                from sklearn.model_selection import train_test_split
                train_idx, temp_idx = train_test_split(
                    indices, test_size=0.4, random_state=config.RANDOM_SEED, stratify=y
                )
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=0.5, random_state=config.RANDOM_SEED, stratify=y[temp_idx]
                )
                
            X_global_test = X_global[test_idx]
            X_local_test = X_local[test_idx]
            X_stellar_test = X_stellar[test_idx]
            y_test = y[test_idx]
            
            # Load predictions
            cnn_data = np.load(predictions_npz)
            y_pred_probs = cnn_data["y_pred_probs"]
            
            # Load model
            print(f"[PostProcess] Instantiating model and loading checkpoint {best_model_pt.name}...")
            model = get_model(
                "astronet",
                global_in_size=config.GLOBAL_BINS,
                local_in_size=config.LOCAL_BINS,
                stellar_in_size=3
            )
            # Force CPU execution for stable post-processing on macOS/Apple Silicon
            device = torch.device("cpu")
            model.load_checkpoint(best_model_pt, device=device)
            
            # Run Grad-CAM
            print("[PostProcess] Generating Grad-CAM heatmaps...")
            x_glob_tensor = torch.tensor(X_global_test, dtype=torch.float32)
            x_loc_tensor = torch.tensor(X_local_test, dtype=torch.float32)
            x_stel_tensor = torch.tensor(X_stellar_test, dtype=torch.float32)
            
            g_cam, l_cam = generate_gradcam(model, x_glob_tensor, x_loc_tensor, x_stel_tensor, device)
            
            print("[PostProcess] Exporting Grad-CAM plots, arrays and metadata...")
            save_gradcam_figures(X_global_test, X_local_test, g_cam, l_cam, y_test, y_pred_probs, config)
            
            # --- Perform Grad-CAM Assertions Validation ---
            print("[PostProcess] Validating Grad-CAM outputs...")
            validation_passed = True
            errors = []
            
            # 1. NaN checks
            if np.isnan(g_cam).any() or np.isnan(l_cam).any():
                validation_passed = False
                errors.append("Heatmaps contain NaN values")
                
            # 2. Dimensions checks
            if g_cam.shape != X_global_test.shape[:2] or l_cam.shape != X_local_test.shape[:2]:
                validation_passed = False
                errors.append(f"Dimensions mismatch: g_cam shape {g_cam.shape} vs input shape {X_global_test.shape}, l_cam shape {l_cam.shape} vs input shape {X_local_test.shape}")
                
            # 3. Output files check
            global_files = list(config.GRADCAM_GLOBAL_DIR.glob("*.png"))
            local_files = list(config.GRADCAM_LOCAL_DIR.glob("*.png"))
            overlay_files = list(config.GRADCAM_OVERLAYS_DIR.glob("*.png"))
            comparison_files = list(config.GRADCAM_COMPARISON_DIR.glob("*.png"))
            numpy_files = list(config.GRADCAM_NUMPY_DIR.glob("*.npz"))
            
            if len(global_files) == 0 or len(local_files) == 0 or len(overlay_files) == 0 or len(comparison_files) == 0:
                validation_passed = False
                errors.append("Visualizations are missing in output directories")
                
            validation_summary = {
                "all_passed": validation_passed,
                "device": str(device),
                "batch_size": int(len(test_idx)),
                "global_cam_shape": list(g_cam.shape),
                "local_cam_shape": list(l_cam.shape),
                "generated_files_count": {
                    "global": len(global_files),
                    "local": len(local_files),
                    "overlays": len(overlay_files),
                    "comparison": len(comparison_files),
                    "numpy": len(numpy_files)
                },
                "errors": errors
            }
            
            validation_path = config.VALIDATION_DIR / "gradcam_validation.json"
            with open(validation_path, "w") as f:
                json.dump(validation_summary, f, indent=4)
            print(f"[PostProcess] Grad-CAM validation report saved to {validation_path.name}")
        else:
            print("[PostProcess] Error: Missing checkpoints, datasets or predictions required for Grad-CAM.")
        
    print("[PostProcess] Post-experiment processing completed successfully!")

if __name__ == "__main__":
    main()
