import streamlit as st
import json
from pathlib import Path
from PIL import Image
import pandas as pd
from utils import inject_custom_css, ROOT_DIR, ARTIFACTS_DIR, EXPLAINABILITY_DIR, GRADCAM_DIR

GRADCAM_OVERLAYS_DIR = GRADCAM_DIR / "overlays"
GRADCAM_COMPARISON_DIR = GRADCAM_DIR / "comparison"

def main():
    inject_custom_css()
    
    st.markdown("""
    <div class="brand-container">
        <div style="font-size: 3rem; line-height: 1;">🛡️</div>
        <div>
            <div class="brand-title">Model Explainability <span class="brand-accent">Engine</span></div>
            <div class="brand-subtitle">Visualizing neural network activations and GBDT Shapley attributions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_global, tab_local, tab_download = st.tabs([
        "Global Interpretability (SHAP)",
        "Local Attribution (SHAP & Grad-CAM)",
        "Download Diagnostics"
    ])
    
    with tab_global:
        st.markdown("### **Global Feature Importance**")
        st.write(
            "Shapley values reflect how much each of the 131 hybrid features (128 CNN deep spatial representation + "
            "3 physical host-star parameters) influences exoplanet classification. Features with high mean absolute "
            "SHAP values exert the largest impact on candidate classification."
        )
        
        c_bar, c_bee = st.columns(2)
        with c_bar:
            st.markdown('<div class="chart-wrap"><div class="chart-title">Global Feature Importance (Bar)</div><div class="chart-subtitle">Mean absolute SHAP value impact</div>', unsafe_allow_html=True)
            img_bar_path = EXPLAINABILITY_DIR / "global_summary.png"
            if img_bar_path.exists():
                st.image(Image.open(img_bar_path), use_container_width=True)
            else:
                st.info("Global summary bar plot not found. Run training first.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_bee:
            st.markdown('<div class="chart-wrap"><div class="chart-title">SHAP Beeswarm Summary Plot</div><div class="chart-subtitle">Feature values color coding attributions</div>', unsafe_allow_html=True)
            img_bee_path = EXPLAINABILITY_DIR / "beeswarm.png"
            if img_bee_path.exists():
                st.image(Image.open(img_bee_path), use_container_width=True)
            else:
                st.info("Beeswarm plot not found. Run training first.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Dependence plot
        st.markdown("### **SHAP Dependence Plot**")
        st.write(
            "SHAP dependence plots analyze non-linear interactions by mapping individual feature values on the X-axis "
            "against their direct SHAP impact on the Y-axis."
        )
        img_dep_path = EXPLAINABILITY_DIR / "dependence_plot.png"
        if img_dep_path.exists():
            st.image(Image.open(img_dep_path), use_container_width=True)
        else:
            st.info("Dependence plot not found.")
            
    with tab_local:
        st.markdown("### **Target Candidate Attributions**")
        
        sample_idx = st.selectbox("Select Test Set Sample Index:", [0, 1, 2, 3], index=0)
        
        # Load explanation json
        exp_json_path = EXPLAINABILITY_DIR / "explanation.json"
        has_explanation = False
        sample_data = {}
        if exp_json_path.exists():
            try:
                with open(exp_json_path, "r") as f:
                    exp_data = json.load(f)
                    if sample_idx < len(exp_data):
                        sample_data = exp_data[sample_idx]
                        has_explanation = True
            except Exception:
                pass
                
        col_list, col_vis = st.columns([4, 6])
        
        with col_list:
            st.markdown("#### **Top 10 Feature Contributions**")
            if has_explanation:
                top_feats = sample_data.get("top_features", [])
                
                rows_html = ""
                for item in top_feats:
                    name = item["feature"]
                    val = item["contribution"]
                    badge_style = "color: #22c55e;" if val > 0 else "color: #ef4444;"
                    rows_html += f"""
                    <tr>
                        <td><code>{name}</code></td>
                        <td style="text-align: right; font-weight: 700; {badge_style}">{val:+.5f}</td>
                    </tr>
                    """
                    
                table_html = f"""
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th style="text-align: right;">SHAP Contribution</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                """
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("No top feature attributions available.")
                
            # Render specific waterfall plot
            st.markdown("#### **SHAP Prediction Waterfall**")
            waterfall_file = EXPLAINABILITY_DIR / f"waterfall_plot_sample_{sample_idx}.png"
            if not waterfall_file.exists() and sample_idx in [1, 2]:
                waterfall_file = EXPLAINABILITY_DIR / f"waterfall_sample_{sample_idx}.png"
                
            if waterfall_file.exists():
                st.image(Image.open(waterfall_file), use_container_width=True)
            else:
                st.info("Sample waterfall plot not found.")
                
        with col_vis:
            st.markdown("#### **CNN Branch Spatial Grad-CAM Overlay**")
            st.write(
                "Grad-CAM tracks backpropagated gradients of target exoplanet candidate scores with respect to activations "
                "of the last convolutional layer. Below are the spatial attribution heatmaps of Global and Local views."
            )
            
            # Load overlay images
            g_overlay_path = GRADCAM_OVERLAYS_DIR / f"overlay_global_sample_{sample_idx}.png"
            l_overlay_path = GRADCAM_OVERLAYS_DIR / f"overlay_local_sample_{sample_idx}.png"
            comparison_grid_path = GRADCAM_COMPARISON_DIR / f"comparison_grid_sample_{sample_idx}.png"
            
            if comparison_grid_path.exists():
                st.image(Image.open(comparison_grid_path), caption=f"Sample {sample_idx} Global & Local Views Comparison Grid", use_container_width=True)
            elif g_overlay_path.exists() and l_overlay_path.exists():
                st.image(Image.open(g_overlay_path), caption=f"Sample {sample_idx} Global View Overlay", use_container_width=True)
                st.image(Image.open(l_overlay_path), caption=f"Sample {sample_idx} Local View Overlay", use_container_width=True)
            else:
                st.info("Grad-CAM visualization overlays not found.")
                
    with tab_download:
        st.markdown("### **Export Explainability Diagnostics**")
        st.write(
            "Download the structured JSON telemetry report containing predictions, probabilities, timestamps, "
            "top contributing features, and full SHAP values for the test partition."
        )
        
        if exp_json_path.exists():
            with open(exp_json_path, "r") as f:
                exp_json_bytes = f.read()
            st.download_button(
                label="Download explanation.json",
                data=exp_json_bytes,
                file_name="explanation.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("No explanation.json file discovered.")

if __name__ == "__main__":
    main()
