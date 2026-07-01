import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import datetime
import json

logger = logging.getLogger("StarSight.Explainability")

def generate_shap_explanations(
    lgb_model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    save_dir: Path
) -> None:
    """
    Computes SHAP explainability values for the LightGBM classifier and saves
    global importance, summary, beeswarm, dependence, waterfall, and explanation JSON.
    
    Args:
        lgb_model: Trained LightGBM booster or LGBMClassifier.
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
    
    # --- 1. Save shap_values.npz ---
    shap_data_path = save_dir / "shap_values.npz"
    np.savez_compressed(
        shap_data_path,
        shap_values=shap_values,
        base_value=expected_value,
        test_features=X_test
    )
    logger.info(f"Saved SHAP explanations arrays to {shap_data_path.resolve()}")
    
    # --- 2. Save feature_importance.csv ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    importance_csv_path = save_dir / "feature_importance.csv"
    df_importance.to_csv(importance_csv_path, index=False)
    logger.info(f"Saved SHAP feature importance to {importance_csv_path.name}")
    
    # --- 3. Save global_summary.png and global_summary.svg ---
    logger.info("Generating SHAP Global Summary (Bar) plots...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title("StarSight Hybrid Model - SHAP Global Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "global_summary.png", dpi=300, transparent=True)
    plt.savefig(save_dir / "global_summary.svg", dpi=300, transparent=True)
    plt.close()
    
    # --- 4. Save beeswarm.png ---
    logger.info("Generating SHAP Beeswarm plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title("StarSight Hybrid Model - SHAP Beeswarm Summary Plot", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "beeswarm.png", dpi=300, transparent=True)
    plt.close()
    
    # --- 5. Save dependence_plot.png ---
    logger.info("Generating SHAP Dependence plot...")
    top_feature_name = df_importance.iloc[0]["feature"]
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature_name, shap_values, X_test_df, show=False)
    plt.title(f"SHAP Dependence Plot - {top_feature_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "dependence_plot.png", dpi=300, transparent=True)
    plt.close()
    
    # --- 6. Save waterfall_sample_1.png and waterfall_sample_2.png ---
    logger.info("Generating specific waterfall sample plots...")
    for idx in [1, 2]:
        if idx < len(X_test):
            exp = shap.Explanation(
                values=shap_values[idx],
                base_values=expected_value,
                data=X_test[idx],
                feature_names=feature_names
            )
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, show=False)
            plt.title(f"StarSight Prediction Waterfall - Sample {idx}", fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(save_dir / f"waterfall_sample_{idx}.png", dpi=300, transparent=True)
            plt.close()
            
    # For backward-compatibility with other scripts, save all samples as well
    for i in range(len(X_test)):
        exp = shap.Explanation(
            values=shap_values[i],
            base_values=expected_value,
            data=X_test[i],
            feature_names=feature_names
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        plt.title(f"StarSight Prediction Waterfall - Sample {i}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / f"waterfall_plot_sample_{i}.png", dpi=150)
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
        
    # --- 7. Save explanation.json ---
    logger.info("Generating explanation.json...")
    # Predict probabilities on test set
    if hasattr(lgb_model, "predict_proba"):
        probs = lgb_model.predict_proba(X_test)
        if probs.ndim == 2:
            probs = probs[:, 1]
    else:
        probs = lgb_model.predict(X_test)  # raw Booster predicts probabilities directly
        
    preds = (probs >= 0.5).astype(int)
    
    explanation_list = []
    timestamp = datetime.datetime.now().isoformat()
    
    for i in range(len(X_test)):
        contributions = {feature_names[j]: float(shap_values[i][j]) for j in range(len(feature_names))}
        # Sort contributions to find top features
        sorted_feats = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_features = [{"feature": name, "contribution": val} for name, val in sorted_feats[:10]]
        
        explanation_list.append({
            "sample_index": i,
            "prediction": int(preds[i]),
            "probability": float(probs[i]),
            "top_features": top_features,
            "feature_contributions": contributions,
            "timestamp": timestamp
        })
        
    with open(save_dir / "explanation.json", "w") as f:
        json.dump(explanation_list, f, indent=4)
        
    # Also save copy to shap_explanations.npz for backward compatibility
    np.savez_compressed(
        save_dir / "shap_explanations.npz",
        shap_values=shap_values,
        base_value=expected_value,
        test_features=X_test
    )
    logger.info("SHAP explanations generated successfully.")
