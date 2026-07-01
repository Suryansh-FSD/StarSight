import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import datetime
from pathlib import Path
from utils import inject_custom_css, load_models, run_pipeline_prediction, get_plotly_layout

def main():
    inject_custom_css()
    
    st.markdown("""
    <div class="brand-container">
        <div style="font-size: 3rem; line-height: 1;">🔮</div>
        <div>
            <div class="brand-title">Transit Prediction <span class="brand-accent">Pipeline</span></div>
            <div class="brand-subtitle">Upload Kepler light curves to extract features and predict candidates</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load PyTorch & LightGBM models
    cnn_model, lgb_clf = load_models()
    if cnn_model is None or lgb_clf is None:
        st.warning("Model checkpoints not fully loaded. Make sure you ran training first!")
        return
        
    col_input, col_results = st.columns([4, 6])
    
    with col_input:
        st.markdown("### **1. Input Ingestion**")
        uploaded_file = st.file_uploader("Upload Light Curve (FITS or CSV)", type=["fits", "csv"])
        
        st.markdown("### **2. Transit Search Parameters**")
        period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=100.0, value=9.52, step=0.01)
        t0 = st.number_input("Transit Epoch (t0 / BJD)", min_value=0.0, max_value=1e5, value=120.35, step=0.1)
        duration = st.number_input("Transit Duration (days)", min_value=0.01, max_value=1.0, value=0.127, step=0.001)
        
        run_prediction = st.button("Execute Vetting Pipeline", use_container_width=True)
        
    with col_results:
        st.markdown("### **3. Vetting Verdict**")
        if uploaded_file is not None and run_prediction:
            with st.spinner("Processing light curve and running model vetting..."):
                try:
                    res = run_pipeline_prediction(uploaded_file, period, t0, duration, cnn_model, lgb_clf)
                    st.session_state.prediction_results = res
                except Exception as e:
                    st.error(f"Error executing pipeline: {e}")
                    return
                    
        if "prediction_results" in st.session_state:
            res = st.session_state.prediction_results
            prob = res["probability"]
            pred = res["prediction"]
            inf_time = res["inference_time_ms"]
            
            # Badge & stats row
            c_badge, c_prob, c_time = st.columns(3)
            with c_badge:
                if pred == 1:
                    st.markdown("""
                    <div class="metric-card" style="text-align: center;">
                        <span class="badge badge-planet">PLANET CANDIDATE</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card" style="text-align: center;">
                        <span class="badge badge-fp">FALSE POSITIVE</span>
                    </div>
                    """, unsafe_allow_html=True)
            with c_prob:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Probability</div>
                    <div class="metric-value" style="color: #ff6f00;">{prob * 100.0:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c_time:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Inference Time</div>
                    <div class="metric-value">{inf_time:.2f} ms</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Expose prediction JSON download
            pred_json = {
                "filename": uploaded_file.name if uploaded_file else "sample",
                "period_days": period,
                "epoch_t0": t0,
                "duration_days": duration,
                "probability": prob,
                "prediction_label": "Planet Candidate" if pred == 1 else "False Positive",
                "inference_time_ms": inf_time,
                "timestamp": res["timestamp"]
            }
            
            st.download_button(
                label="Download Prediction JSON",
                data=json.dumps(pred_json, indent=4),
                file_name="star_sight_prediction.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("Upload a file and click 'Execute Vetting Pipeline' to view predictions.")
            
    # Visualizations section
    if "prediction_results" in st.session_state:
        res = st.session_state.prediction_results
        
        st.markdown("---")
        st.markdown("### **4. Visual Attributions**")
        
        tab_detrend, tab_folded = st.tabs(["Stellar Detrending", "Phase Folded Views"])
        
        with tab_detrend:
            # Clean vs trend plot
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=res["time"], y=res["flux"], mode="markers",
                marker=dict(size=2, color="#71717a"), name="PDCSAP Flux"
            ))
            fig1.add_trace(go.Scatter(
                x=res["time_flat"], y=res["trend_flux"], mode="lines",
                line=dict(color="#ff6f00", width=1.5), name="Biweight Trend"
            ))
            fig1.update_layout(
                title="Stellar Detrending Model",
                xaxis_title="Time (BJD)",
                yaxis_title="Normalized Flux",
                **get_plotly_layout()
            )
            
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
            
            # Flattened plot
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=res["time_flat"], y=res["flux_flat"], mode="markers",
                marker=dict(size=2, color="#ff6f00"), name="Flattened Flux"
            ))
            fig2.update_layout(
                title="Flattened Light Curve (Stellar Variability Removed)",
                xaxis_title="Time (BJD)",
                yaxis_title="Relative Flux Offset",
                **get_plotly_layout()
            )
            
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            
        with tab_folded:
            c_g, c_l = st.columns(2)
            with c_g:
                st.markdown('<div class="chart-wrap"><div class="chart-title">Global View (2001 Bins)</div><div class="chart-subtitle">Binned full orbital phase [-0.5, 0.5]</div>', unsafe_allow_html=True)
                fig_g = go.Figure()
                fig_g.add_trace(go.Scatter(
                    x=np.linspace(-0.5, 0.5, len(res["global_view"])),
                    y=res["global_view"],
                    mode="lines", line=dict(color="#ff6f00", width=1.5)
                ))
                fig_g.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260,
                    **get_plotly_layout()
                )
                st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_l:
                st.markdown('<div class="chart-wrap"><div class="chart-title">Local View (201 Bins)</div><div class="chart-subtitle">High-resolution zoomed transit window</div>', unsafe_allow_html=True)
                fig_l = go.Figure()
                fig_l.add_trace(go.Scatter(
                    x=np.linspace(-0.5, 0.5, len(res["local_view"])),
                    y=res["local_view"],
                    mode="lines", line=dict(color="#ff6f00", width=1.5)
                ))
                fig_l.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260,
                    **get_plotly_layout()
                )
                st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
