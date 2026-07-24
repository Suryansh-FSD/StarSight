import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import lightgbm as lgb
import pickle
import os
import json
import datetime
from pathlib import Path
from astropy.io import fits
import lightkurve as lk
from scipy.interpolate import interp1d
from scipy.stats import sigmaclip

# Resolve base directories dynamically relative to the script location
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
EXPLAINABILITY_DIR = ARTIFACTS_DIR / "explainability"
GRADCAM_DIR = ARTIFACTS_DIR / "gradcam"
SUMMARIES_DIR = ARTIFACTS_DIR / "summaries"

# Style theme configuration (Space Dark / ISRO Orange Accents)
BG_COLOR = "#0b0c10"
CARD_BG = "#1f2833"
TEXT_COLOR = "#c5c6c7"
ACCENT_COLOR = "#ff6f00"  # ISRO Orange
ACCENT_HOVER = "#e65100"
BORDER_COLOR = "#2a3543"

def inject_custom_css():
    """
    Injects custom CSS to style the Streamlit dashboard with a premium,
    dark space theme, ISRO orange accents, custom scrollbars, and styled cards.
    """
    css = f"""
    <style>
    /* Global Background and Fonts */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: {BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    
    /* Header Hide */
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    .block-container {{
        padding: 2rem 2.5rem 3rem !important;
        max-width: 1360px !important;
    }}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {BG_COLOR};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {BORDER_COLOR};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {ACCENT_COLOR};
    }}
    
    /* Custom Cards and Containers */
    .metric-card {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 10px !important;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: #8b9bb4;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.2rem;
        letter-spacing: -0.02em;
    }}
    .metric-delta {{
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.4rem;
        color: {ACCENT_COLOR};
    }}
    
    .chart-wrap {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 1.2rem 1.2rem 0.6rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 1.25rem;
    }}
    .chart-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
        border-left: 3px solid {ACCENT_COLOR};
        padding-left: 8px;
        margin-bottom: 0.2rem;
    }}
    .chart-subtitle {{
        font-size: 0.75rem;
        color: #8b9bb4;
        margin-bottom: 0.8rem;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #0c0f14 !important;
        border-right: 1px solid {BORDER_COLOR} !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {ACCENT_COLOR} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }}
    .stButton>button:hover {{
        background-color: {ACCENT_HOVER} !important;
        box-shadow: 0 0 10px {ACCENT_COLOR} !important;
        transform: translateY(-1px) !important;
    }}
    
    /* Table Styling */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.82rem;
        background-color: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        overflow: hidden;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.75rem 0.9rem;
        color: #8b9bb4;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid {BORDER_COLOR};
        background-color: #161e27;
    }}
    .data-table td {{
        padding: 0.7rem 0.9rem;
        color: #ffffff;
        border-bottom: 1px solid {BORDER_COLOR};
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    
    /* Status Badges */
    .badge {{
        display: inline-block;
        padding: 2px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .badge-planet {{
        color: #ff9100;
        background: rgba(255, 145, 0, 0.15);
        border: 1px solid rgba(255, 145, 0, 0.3);
    }}
    .badge-fp {{
        color: #ff3d00;
        background: rgba(255, 61, 0, 0.15);
        border: 1px solid rgba(255, 61, 0, 0.3);
    }}
    
    /* Banner/Brand Header */
    .brand-container {{
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid {BORDER_COLOR};
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }}
    .brand-title {{
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
    }}
    .brand-accent {{
        color: {ACCENT_COLOR};
    }}
    .brand-subtitle {{
        font-size: 0.85rem;
        color: #8b9bb4;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def get_plotly_layout(title_text="", is_dark=True):
    """
    Returns consistent themed layout dictionary for Plotly charts.
    """
    text_col = "#a1a1aa" if is_dark else "#71717a"
    grid_col = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.04)"
    
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=text_col, size=11),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(
            gridcolor=grid_col,
            zerolinecolor=grid_col,
            tickfont=dict(size=10, color=text_col),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=grid_col,
            zerolinecolor=grid_col,
            tickfont=dict(size=10, color=text_col),
            showgrid=True,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=text_col, size=10)
        )
    )

def render_metric_card(label, value, delta=None):
    """
    Renders a styled dark-theme metric card.
    """
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """
    Loads PyTorch CNN model (AstroNet) and LightGBMClassifier wrapper.
    """
    cnn_path = MODELS_DIR / "final_model.pt"
    lgb_path = MODELS_DIR / "lightgbm.pkl"
    
    cnn_model = None
    lgb_clf = None
    
    if cnn_path.exists():
        try:
            # We import AstroNet locally so it can load the state_dict
            from src.models.astronet import AstroNet
            import src.config as config
            cnn_model = AstroNet(
                global_in_size=config.GLOBAL_BINS,
                local_in_size=config.LOCAL_BINS,
                stellar_in_size=3
            )
            cnn_model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
            cnn_model.eval()
        except Exception as e:
            st.error(f"Error loading CNN model: {e}")
            
    if lgb_path.exists():
        try:
            from src.models.lightgbm_classifier import LightGBMClassifier
            lgb_clf = LightGBMClassifier()
            lgb_clf.load(lgb_path)
        except Exception as e:
            st.error(f"Error loading LightGBM model: {e}")
            
    return cnn_model, lgb_clf

def run_pipeline_prediction(uploaded_file, period, t0, duration, cnn_model, lgb_clf):
    """
    Executes light curve preprocessing, folding, feature binning, CNN feature extraction,
    and LightGBM prediction on the uploaded file.
    """
    import src.config as config
    from scipy.interpolate import interp1d
    
    # 1. Read light curve
    filename = uploaded_file.name
    try:
        if filename.endswith(".fits"):
            with fits.open(uploaded_file) as hdul:
                # Primary header stellar parameters extraction
                hdr = hdul[0].header
                teff = float(hdr.get("TEFF", 5778.0))
                radius = float(hdr.get("RADIUS", 1.0))
                logg = float(hdr.get("LOGG", 4.43))
                
                # Get table data from extension 1
                tb = hdul[1].data
                time = np.asarray(tb["TIME"], dtype=np.float64)
                flux = np.asarray(tb["PDCSAP_FLUX"], dtype=np.float64)
        else:
            # CSV upload format: time, flux, stellar parameters can be read if present
            df = pd.read_csv(uploaded_file)
            time = df["time"].values
            flux = df["flux"].values
            teff = float(df.get("teff", [5778.0])[0])
            radius = float(df.get("radius", [1.0])[0])
            logg = float(df.get("logg", [4.43])[0])
    except Exception as e:
        raise ValueError(f"Error parsing uploaded file format: {e}")
        
    # Clean NaNs
    nan_mask = np.isnan(time) | np.isnan(flux)
    time = time[~nan_mask]
    flux = flux[~nan_mask]
    
    # Sigmaclip outliers (asymmetric: +5 std / -20 std)
    median_flux = np.median(flux)
    std_flux = np.std(flux)
    mask = (flux >= (median_flux - 20.0 * std_flux)) & (flux <= (median_flux + 5.0 * std_flux))
    time_clean = time[mask]
    flux_clean = flux[mask]
    
    # Normalization (relative offset) and biweight detrending using Lightkurve wrapper
    lc = lk.LightCurve(time=time_clean, flux=flux_clean)
    cadences = int(np.round(0.5 / np.nanmedian(np.diff(time_clean))))
    if cadences % 2 == 0:
        cadences += 1
    flat_lc, trend_lc = lc.flatten(window_length=max(3, cadences), return_trend=True)
    
    # Phase fold centering at Phase 0.0
    time_flat = flat_lc.time.value
    flux_flat = flat_lc.flux.value
    
    # Calculate phase folded
    phase = ((time_flat - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux_flat[sort_idx]
    
    # Binning Global view (2001 bins)
    global_bins = config.GLOBAL_BINS if hasattr(config, "GLOBAL_BINS") else 2001
    bin_edges_g = np.linspace(-0.5, 0.5, global_bins + 1)
    bin_centers_g = 0.5 * (bin_edges_g[:-1] + bin_edges_g[1:])
    bin_indices_g = np.digitize(phase_sorted, bin_edges_g) - 1
    global_view = np.zeros(global_bins)
    for i in range(global_bins):
        m = (bin_indices_g == i)
        global_view[i] = np.median(flux_sorted[m]) if np.any(m) else np.nan
        
    nan_m = np.isnan(global_view)
    if np.any(nan_m):
        x_no = bin_centers_g[~nan_m]
        y_no = global_view[~nan_m]
        f_interp = interp1d(x_no, y_no, kind='linear', fill_value='extrapolate')
        global_view[nan_m] = f_interp(bin_centers_g[nan_m])
    global_view -= np.median(global_view)
    
    # Binning Local view (201 bins)
    local_bins = config.LOCAL_BINS if hasattr(config, "LOCAL_BINS") else 201
    half_width = min(2.0 * duration / period, 0.5)
    bin_edges_l = np.linspace(-half_width, half_width, local_bins + 1)
    bin_centers_l = 0.5 * (bin_edges_l[:-1] + bin_edges_l[1:])
    bin_indices_l = np.digitize(phase_sorted, bin_edges_l) - 1
    local_view = np.zeros(local_bins)
    for i in range(local_bins):
        m = (bin_indices_l == i)
        local_view[i] = np.median(flux_sorted[m]) if np.any(m) else np.nan
        
    nan_m_l = np.isnan(local_view)
    if np.any(nan_m_l):
        if np.all(nan_m_l):
            f_g = interp1d(phase_sorted, flux_sorted, kind='linear', fill_value='extrapolate')
            local_view = f_g(bin_centers_l)
        else:
            x_no = bin_centers_l[~nan_m_l]
            y_no = local_view[~nan_m_l]
            f_interp = interp1d(x_no, y_no, kind='linear', fill_value='extrapolate')
            local_view[nan_m_l] = f_interp(bin_centers_l[nan_m_l])
    local_view -= np.median(local_view)
    
    # 2. Run CNN Penultimate features
    device = torch.device("cpu")
    t_global = torch.tensor(global_view, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t_local = torch.tensor(local_view, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    stellar_params = np.array([teff, radius, logg], dtype=np.float32)
    t_stellar = torch.tensor(stellar_params, dtype=torch.float32).unsqueeze(0).to(device)
    
    import time as timer_lib
    inference_start = timer_lib.time()
    with torch.no_grad():
        embedding = cnn_model.extract_features(t_global, t_local, t_stellar).cpu().numpy().flatten()
        
    # Combine with stellar properties
    feature_vector = np.concatenate([embedding, stellar_params]).reshape(1, -1)
    
    # 3. Predict GBDT classification
    prob = float(lgb_clf.predict_proba(feature_vector)[0])
    pred = int(prob >= 0.5)
    inf_time = (timer_lib.time() - inference_start) * 1000.0 # ms
    
    return {
        "time": time_clean,
        "flux": flux_clean,
        "time_flat": time_flat,
        "flux_flat": flux_flat,
        "trend_flux": trend_lc.flux.value,
        "phase_folded": phase_sorted,
        "flux_folded": flux_sorted,
        "global_view": global_view,
        "local_view": local_view,
        "stellar_params": {
            "teff": teff,
            "radius": radius,
            "logg": logg
        },
        "cnn_embedding": embedding,
        "probability": prob,
        "prediction": pred,
        "inference_time_ms": inf_time,
        "timestamp": datetime.datetime.now().isoformat()
    }
