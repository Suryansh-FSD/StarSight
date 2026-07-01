import streamlit as st
from utils import inject_custom_css

def main():
    inject_custom_css()
    
    st.markdown("""
    <div class="brand-container">
        <div style="font-size: 3rem; line-height: 1;">📝</div>
        <div>
            <div class="brand-title">About <span class="brand-accent">StarSight</span></div>
            <div class="brand-subtitle">Project architecture, technology stack, and credentials</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Left Column: Project Metadata / Technology Stack
    # Right Column: Architecture & Pipelines
    col_left, col_right = st.columns([5, 5])
    
    with col_left:
        st.markdown("### **Project Credentials**")
        st.markdown("""
        * **Project Name:** StarSight (Exoplanet Vetting Engine)
        * **Author/Developer:** Suryansh Dixit (FSD Lead)
        * **Repository:** [Suryansh-FSD/StarSight](https://github.com/Suryansh-FSD/StarSight)
        * **License:** Apache License 2.0
        """)
        
        st.markdown("### **Technology Stack**")
        st.markdown("""
        The entire StarSight software suite is built using standard open-source scientific tools:
        
        | Component | Technology | Role |
        | :--- | :--- | :--- |
        | **Language** | Python 3.12 / 3.13 | Core programming interface |
        | **Deep Learning** | PyTorch (`torch`) | Neural network design & parameter optimization |
        | **Downstream GBDT** | LightGBM | Multi-layered candidate feature classification |
        | **Astronomical Utilities** | Lightkurve & Astropy | FITS headers parsing, Kepler target downloads, and detrending |
        | **Explainability** | SHAP & Grad-CAM | Visual attributions and Shapley values calculation |
        | **Dashboard App** | Streamlit | Responsive visual user interface |
        | **Plotting Engine** | Plotly | Dynamic, interactive charts |
        """)
        
    with col_right:
        st.markdown("### **Architectural Workflow**")
        st.write(
            "StarSight is engineered with a modular directory structure ensuring single-responsibility modules "
            "and scientific reproducibility. The core pipeline maps as follows:"
        )
        
        st.markdown("""
        1. **`src/config.py`**: Holds path variables, training hyperparameters, and hardware device definitions.
        2. **`src/experiment/experiment.py`**: Life-cycle orchestrator directing train/val/test data splits, checkpointing, and GBDT model fitting.
        3. **`src/models/astronet.py`**: Employs late-fusion CNN feature extraction across binned Global (2001 bins) and Local (201 bins) views.
        4. **`src/models/lightgbm_classifier.py`**: High-performance GBDT wrapper exposing standardized `train`, `predict`, and `save/load` methods.
        5. **`src/models/gradcam.py`**: Computes spatial CNN branch activation importances.
        6. **`src/models/explainability.py`**: Implements game-theoretic SHAP local and global attributions.
        """)
        
        st.markdown("### **Pipeline Ingestion & Target Selection**")
        st.write(
            "Target Kepler objects of interest (e.g. Kepler-10, Kepler-90) are fetched from the STScI MAST archive. "
            "PDCSAP flux observations are preprocessed, folded centered at Phase 0.0, and converted into fixed-size "
            "feature representations for classifier prediction."
        )

if __name__ == "__main__":
    main()
