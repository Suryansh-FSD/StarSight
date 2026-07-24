import streamlit as st
import pandas as pd
import json
from pathlib import Path
from utils import inject_custom_css, render_metric_card, ROOT_DIR, ARTIFACTS_DIR, METRICS_DIR, SUMMARIES_DIR

def main():
    inject_custom_css()
    
    # 1. Header & Logo
    st.markdown("""
    <div class="brand-container">
        <div style="font-size: 3rem; line-height: 1;">🌌</div>
        <div>
            <div class="brand-title">StarSight <span class="brand-accent">Exoplanet Vetting Engine</span></div>
            <div class="brand-subtitle">AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Project Overview & Supported Missions
    col_left, col_right = st.columns([7, 3])
    with col_left:
        st.markdown("### **Project Description**")
        st.write(
            "StarSight is an advanced end-to-end scientific pipeline designed to detect periodic transit signals "
            "of exoplanets from raw astronomical light curves. By leveraging a state-of-the-art dual-branch "
            "1D Convolutional Neural Network (AstroNet-inspired) combined with a downstream LightGBM GBDT classifier, "
            "the pipeline filters stellar rotational variability, fits transit parameters, extracts deep feature "
            "embeddings, and performs high-confidence candidate classification."
        )
        
        st.markdown("### **Supported Space Missions**")
        st.markdown("""
        * 🚀 **Kepler Mission (K1/K2):** Full PDCSAP flux ingestion, detrending, and candidate vetting.
        * 🛰️ **TESS Mission (Transiting Exoplanet Survey Satellite):** (Planned telemetry scaling for short-cadence profiles).
        """)
        
    with col_right:
        st.markdown("### **Current Model Status**")
        st.markdown("""
        <div class="metric-card" style="margin-bottom: 0px;">
            <div class="metric-label" style="color: #ff6f00; font-weight: 700;">Active Model</div>
            <div class="metric-value" style="font-size: 1.45rem;">Hybrid CNN + GBDT</div>
            <div style="font-size: 0.78rem; color: #8b9bb4; margin-top: 0.5rem; line-height: 1.4;">
                • AstroNet 1D CNN Encoder<br>
                • 128 Penultimate Embeddings<br>
                • LightGBM Downstream Classifier
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # 3. Pipeline Flow Diagram
    st.markdown("---")
    st.markdown("### **StarSight Execution Pipeline**")
    
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #1f2833; border: 1px solid #2a3543; border-radius: 8px; padding: 1.5rem; margin-top: 1rem;">
        <div style="flex: 1; text-align: center;">
            <div style="font-size: 1.5rem;">📥</div>
            <strong style="color: #ffffff; font-size: 0.85rem;">1. Data Acquisition</strong>
            <div style="font-size: 0.72rem; color: #8b9bb4; margin-top: 4px;">FITS download from MAST</div>
        </div>
        <div style="font-size: 1.5rem; color: #ff6f00;">➜</div>
        <div style="flex: 1; text-align: center;">
            <div style="font-size: 1.5rem;">🧹</div>
            <strong style="color: #ffffff; font-size: 0.85rem;">2. Preprocessing</strong>
            <div style="font-size: 0.72rem; color: #8b9bb4; margin-top: 4px;">Sigma clipping & detrending</div>
        </div>
        <div style="font-size: 1.5rem; color: #ff6f00;">➜</div>
        <div style="flex: 1; text-align: center;">
            <div style="font-size: 1.5rem;">🔎</div>
            <strong style="color: #ffffff; font-size: 0.85rem;">3. BLS Search</strong>
            <div style="font-size: 0.72rem; color: #8b9bb4; margin-top: 4px;">Period & epoch discovery</div>
        </div>
        <div style="font-size: 1.5rem; color: #ff6f00;">➜</div>
        <div style="flex: 1; text-align: center;">
            <div style="font-size: 1.5rem;">📊</div>
            <strong style="color: #ffffff; font-size: 0.85rem;">4. Feature Binning</strong>
            <div style="font-size: 0.72rem; color: #8b9bb4; margin-top: 4px;">Global & Local phase folding</div>
        </div>
        <div style="font-size: 1.5rem; color: #ff6f00;">➜</div>
        <div style="flex: 1; text-align: center;">
            <div style="font-size: 1.5rem;">🧠</div>
            <strong style="color: #ffffff; font-size: 0.85rem;">5. Deep Learning</strong>
            <div style="font-size: 0.72rem; color: #8b9bb4; margin-top: 4px;">Penultimate late fusion</div>
        </div>
        <div style="font-size: 1.5rem; color: #ff6f00;">➜</div>
        <div style="flex: 1; text-align: center;">
            <div style="font-size: 1.5rem;">🌳</div>
            <strong style="color: #ffffff; font-size: 0.85rem;">6. LightGBM GBDT</strong>
            <div style="font-size: 0.72rem; color: #8b9bb4; margin-top: 4px;">Candidate prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Latest Metrics Cards
    st.markdown("---")
    st.markdown("### **Latest Hybrid Model Metrics**")
    
    # Load evaluation metrics
    metrics_path = METRICS_DIR / "lightgbm_metrics.csv"
    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            metrics = df.iloc[0].to_dict()
        except Exception:
            metrics = {}
    else:
        metrics = {}
        
    acc = metrics.get("accuracy", 1.0)
    prec = metrics.get("precision", 1.0)
    rec = metrics.get("recall", 1.0)
    f1 = metrics.get("f1_score", 1.0)
    auc = metrics.get("roc_auc", 1.0)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Accuracy", f"{acc:.4f}", "Hybrid model accuracy")
    with c2:
        render_metric_card("Precision", f"{prec:.4f}", "True candidate prediction rate")
    with c3:
        render_metric_card("Recall", f"{rec:.4f}", "Candidate recall sensitivity")
    with c4:
        render_metric_card("F1-Score", f"{f1:.4f}", "Balanced precision/recall")
    with c5:
        render_metric_card("ROC-AUC", f"{auc:.4f}", "Class separability score")

if __name__ == "__main__":
    main()
