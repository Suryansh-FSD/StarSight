import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger("StarSight.Explainability")

def generate_shap_explanations(
    lgb_model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    save_dir: Path
) -> None:
    """
    Computes SHAP explainability values for the LightGBM classifier and saves
    global importance, summary, force, and waterfall plots to disk.
    
    Args:
        lgb_model: Trained LightGBM model.
        X_train: Feature embeddings array for the training partition.
        X_test: Feature embeddings array for the test partition.
        save_dir: Path directory where explainability plots are persisted.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define descriptive feature names
    feature_names = [f"cnn_dim_{i}" for i in range(128)] + ["stellar_teff", "stellar_radius", "stellar_logg"]
    
    logger.info("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(lgb_model)
    
    # Compute SHAP values on the test set
    logger.info("Computing SHAP values on the test set...")
    shap_values_raw = explainer.shap_values(X_test)
    
    # Handle list-based output for binary classifiers in different SHAP/LightGBM versions
    if isinstance(shap_values_raw, list):
        # Index 1 contains SHAP values for the positive class (planet candidate probability)
        shap_values = shap_values_raw[1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        # Single output array
        shap_values = shap_values_raw
        expected_value = explainer.expected_value
        
    # Convert test set to DataFrame with proper feature names for plotting
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # 1. Global Feature Importance (Bar Plot)
    logger.info("Generating SHAP Global Feature Importance plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title("StarSight Hybrid Model - SHAP Global Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    bar_path = save_dir / "global_feature_importance.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()
    
    # 2. Summary Beeswarm Plot
    logger.info("Generating SHAP Beeswarm Summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title("StarSight Hybrid Model - SHAP Summary Beeswarm Plot", fontsize=12, fontweight="bold")
    plt.tight_layout()
    summary_path = save_dir / "summary_beeswarm_plot.png"
    plt.savefig(summary_path, dpi=150)
    plt.close()
    
    # 3. Individual explanations: Force and Waterfall plots for every sample in the test set
    logger.info("Generating sample-by-sample SHAP plots...")
    for i in range(len(X_test)):
        # Construct Explanation object for waterfall plot
        exp = shap.Explanation(
            values=shap_values[i],
            base_values=expected_value,
            data=X_test[i],
            feature_names=feature_names
        )
        
        # Save Waterfall Plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        plt.title(f"StarSight Prediction Waterfall - Sample {i}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        waterfall_path = save_dir / f"waterfall_plot_sample_{i}.png"
        plt.savefig(waterfall_path, dpi=150)
        plt.close()
        
        # Save Force Plot (HTML format)
        force_plot = shap.force_plot(
            expected_value,
            shap_values[i],
            X_test[i],
            feature_names=feature_names,
            matplotlib=False
        )
        force_path = save_dir / f"force_plot_sample_{i}.html"
        shap.save_html(str(force_path), force_plot)
        
    # Save raw explanations array to disk
    shap_data_path = save_dir / "shap_explanations.npz"
    np.savez_compressed(
        shap_data_path,
        shap_values=shap_values,
        base_value=expected_value,
        test_features=X_test
    )
    logger.info(f"Saved SHAP explanations arrays to {shap_data_path.resolve()}")
