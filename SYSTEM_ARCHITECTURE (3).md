# **StarSight**

## ***AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves***

**Document Reference:** ISRO-BAH-2026-EXO-ADS-ARCH-001  
**Classification:** Technical Design Document  
**Target Event:** Bharatiya Antariksh Hackathon 2026  
**System Name:** StarSight (Exoplanet Automated Detection System)

---

## **1. Executive Summary**

StarSight is an AI-powered system for detecting exoplanets from noisy astronomical light curves. The system downloads Kepler space telescope observations, preprocesses the raw time-series, detects potential transit events using Box Least Squares (BLS), generates standardized Global and Local Views, and classifies candidates using a custom hybrid deep learning and gradient boosted decision tree architecture. The pipeline processes inputs via a Dual-Branch Convolutional Neural Network (CNN) model (AstroNet format) to extract penultimate 128-dimensional spatial feature representations, concatenates them with physical host-star attributes, and feeds the resulting 131-dimensional feature vector into a LightGBM classifier. The pipeline automatically calculates SHAP (SHapley Additive exPlanations) values using a TreeExplainer to explain individual and global GBDT predictions, and generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps on the CNN encoder to visually identify the specific light curve regions driving CNN feature representations. Explainability diagnostics are saved inside dedicated directories in PNG, SVG, and raw array formats. The architecture is optimized for rapid local and cloud execution, scientific correctness, and validation.

---

## **2. Problem Statement**

Automated exoplanet detection pipelines operating on space-based photometric datasets face persistent data challenges that degrade naive machine learning configurations:

1. **Extreme Class Imbalance:** Labeled Threshold Crossing Event (TCE) datasets exhibit massive structural skews. Non-planetary noise and astrophysical false alarms outnumber valid planet candidates, causing standard classifiers to stall or default to majority-class predictions.  
2. **Multi-Faceted Noise Profiles:** Space telescope telemetry is obscured by high-frequency instrumental artifacts (pointing jitters, thermal drifts, aperture spills), localized cosmic ray pixel spikes, and natural stellar variability (pulsations, flares, starspots).  
3. **Astrophysical Confounding Modulations:** Grazing or detached Eclipsing Binaries (EBs) and background star light dilutions in crowded fields yield light curve geometries that closely mimic the shallow, periodic flux drops of genuine transiting exoplanets.  
4. **Periodicity Identification:** A robust Box Least Squares (BLS) search is required to estimate the orbital period and transit epoch before phase folding.

---

## **3. Objectives**

### **Technical Objectives**

* **Optimized Execution Pipeline:** Build an integrated data parsing and training stream optimized for fast prototyping and low computational overhead.  
* **Leakage-Protected Partitioning:** Implement dataset splits that enforce group-stratified splits by host star KIC ID to prevent temporal leakage across train/validation/test pools.  
* **Hybrid Classification Architecture:** Deploy a late-fusion GBDT classifier on top of deep spatial CNN representations to optimize decision boundaries on highly imbalanced inputs.

### **Scientific Objectives**

* **Stellar Context Grounding:** Incorporate host-star physical properties ($R_*$, effective temperature $T_{eff}$, and surface gravity $\log g$) directly into the final classification head to ground predictions in real stellar physics.  
* **Systematics Rectification:** Process raw light curves by applying NaN removal, asymmetric outlier sigma clipping, biweight detrending, and normalization to isolate the transit signal cleanly.  
* **Strict Interpretability & Reproducibility:** Ensure all model predictions are interpreted dynamically via Shapley feature attributions (global and local) and Grad-CAM spatial heatmaps (Global and Local views), and make all random operations deterministic and governed by a centralized seed.

### **Hackathon & Deployment Scope**

* **Local & Cloud Compatibility:** Ensure the entire codebase runs identically on both a local development environment (leveraging Apple Silicon GPU/MPS) and Google Colab (utilizing CUDA GPU acceleration with Google Drive persistent storage).

---

## **4. System Workflow**

The StarSight end-to-end data processing workflow is organized into six sequential phases:  
**NASA Kepler Archive → Download FITS Files → PDCSAP Flux Extraction → Cleaning (NaN Removal, Asymmetric Sigma Clipping, Normalization, Biweight Detrending) → Box Least Squares (BLS) Periodogram Sweep → Phase Folding → AstroNet Global & Local Views Generation → Dual-Branch CNN Feature Extraction (Global + Local branches) → Penultimate Embedding Extraction (128-dim) → Concatenation with Stellar Parameters (131-dim) → Downstream LightGBM Classification → SHAP Interpretability Attributions & Grad-CAM CNN Spatial Attributions → Planet Candidate Probability Output + Explanations.**

```
graph TD
A[NASA Kepler Archive] --> B[Download FITS Files]
B --> C[PDCSAP Flux Extraction]
C --> D[Cleaning & Detrending]
D --> E[Box Least Squares Periodogram]
E --> F[Phase Folding]
F --> G[Global View (2001 bins)]
F --> H[Local View (201 bins)]
G --> I[Dual 1D CNN]
H --> I
I --> J[Extract 128-dim Penultimate Representation]
J --> K[Concatenate with Stellar Parameters -> 131-dim]
K --> L[LightGBM GBDT Classifier]
L --> N[SHAP Explainability Analyzer]
I --> Q[Grad-CAM Heatmap Generator]
L --> M[Candidate Probability Output]
N --> O[Global Importance & Beeswarm Plots]
N --> P[Waterfall & Force plots per prediction]
Q --> R[Global & Local Spatial Heatmaps & Overlays]
```

---

## **5. Software Architecture**

The granular breakdown of each software module:

### **Module 1: Telemetry Data Ingestion (data_acquisition)**
* **Purpose:** Stream raw multi-dimensional satellite telemetry files and isolate uncorrected target light curves.  
* **Responsibilities:** Parse FITS files; pull spatial headers; compute integrated flux counts.  
* **Inputs:** Raw space archive `.fits` target pixel files.  
* **Outputs:** DataFrames containing Time ($t$) and PDCSAP Flux.  
* **Python Libraries:** `astropy.io.fits`, `lightkurve`, `numpy`, `pandas`.  
* **Folder Location:** `data/raw/`  

### **Module 2: Science Data Partitioning Engine (datasets)**
* **Purpose:** Partition, structure, and serve data inputs under strict scientific cross-validation rules.  
* **Responsibilities:** Enforce group-stratified splitting by unique target host star identifier to eliminate data leakage.  
* **Inputs:** Time-series binned arrays; tabular host-star physical properties.  
* **Outputs:** Train, Validation, and Test PyTorch DataLoaders.  
* **Python Libraries:** `pandas`, `scikit-learn` (`train_test_split`).  
* **Folder Location:** `data/processed/`  

### **Module 3: Systematics Rectification Core (preprocessing)**
* **Purpose:** Cleanse signal data for downstream classification while preserving physical transit geometric shapes.  
* **Responsibilities:** Apply asymmetric outlier clipping ($+5\sigma$ / $-20\sigma$), perform windowed biweight detrending (0.5 day window) to isolate stellar rotation, and normalize flux.  
* **Inputs:** FITS target pixel files and light curves.  
* **Outputs:** Preprocessed flux timelines.  
* **Python Libraries:** `scipy`, `lightkurve`.  
* **Folder Location:** `data/processed/`  

### **Module 4: Periodicity & Folding Processor (transit_search)**
* **Purpose:** Run Box Least Squares (BLS) to estimate the most likely orbital period and transit epoch before phase folding.  
* **Responsibilities:** Sweep frequency spaces for period ($P$) and epoch ($T_0$); stack continuous timelines; fold and center light curves at phase $0.0$.  
* **Inputs:** Corrected flux timelines.  
* **Outputs:** Phase-folded light curve data.  
* **Python Libraries:** `astropy.timeseries.BoxLeastSquares`, `numpy`.  
* **Folder Location:** `results/`  

### **Module 5: AstroNet Feature Binning Engine (astronet_features)**
* **Purpose:** Bin phase-folded curves and extract physical properties to generate final multi-modal vectors.  
* **Responsibilities:** Bin the full folded period into a macro-scale Global View (2001 bins) and zoom in on the transit event to generate a Local View (201 bins); pull stellar parameters ($T_{eff}$, $R_*$, $\log g$) from FITS primary headers.  
* **Inputs:** Folded timelines and raw FITS headers.  
* **Outputs:** Final unified dataset package (`final_dataset.npz`).  
* **Python Libraries:** `numpy`.  
* **Folder Location:** `data/processed/`

### **Module 6: Trimodal Deep Representation Engine (astronet)**
* **Purpose:** Extract spatial feature embeddings from multi-resolution phase-folded views.  
* **Responsibilities:** Process Global Views via 1D Convolutional layers, process Local Views via parallel 1D Convolutional layers, concatenate deep representations, and extract a 128-dimensional penultimate embedding vector.  
* **Inputs:** Global and Local binned arrays, tabular stellar parameters.  
* **Outputs:** 128-dimensional spatial feature representation.  
* **Python Libraries:** `torch`.  
* **Folder Location:** `src/models/`  

### **Module 7: Downstream Boosting Classifier Core (lightgbm)**
* **Purpose:** Perform final exoplanet candidate classification using gradient boosted decision trees.  
* **Responsibilities:** Concatenate the 128-dimensional CNN representation with 3 physical host-star properties ($T_{eff}$, $R_*$, $\log g$), train a leaf-wise LightGBM GBDT classifier, and save the model configuration.  
* **Inputs:** 131-dimensional combined feature vectors.  
* **Outputs:** Continuous planet candidate probability in the range $[0, 1]$.  
* **Python Libraries:** `lightgbm`.  
* **Folder Location:** `src/experiment/` and `artifacts/models/`

### **Module 8: Explainable AI Engine - SHAP (explainability)**
* **Purpose:** Provide transparent downstream GBDT model transparency diagnostic plots and attribution arrays.  
* **Responsibilities:** Fit `shap.TreeExplainer` on the trained LightGBM model, extract global importance bar plots, beeswarm summary plots, waterfall attribution curves, and interactive force plots for target predictions.  
* **Inputs:** LightGBM booster configuration, training and test embeddings.  
* **Outputs:** Interpretability PNG/HTML diagnostics and `shap_explanations.npz` raw arrays.  
* **Python Libraries:** `shap`, `matplotlib`.  
* **Folder Location:** `src/models/` and `artifacts/explainability/`

### **Module 9: Explainable AI Engine - CNN Grad-CAM (gradcam)**
* **Purpose:** Extract spatial feature attributions from the CNN encoder to visually identify the specific light curve regions driving CNN feature representations.  
* **Responsibilities:** Extract class activation maps (Grad-CAM) for both the Global and Local 1D CNN branches of the AstroNet model. Hooks dynamically into the last convolutional layers to record activations and gradients during backward sweeps. Generates raw signals, normalized heatmaps, overlays, and publication-quality comparisons of Global/Local views in PNG and SVG formats.
* **Inputs:** AstroNet model instance, best model checkpoint, test binned arrays, and target prediction labels.
* **Outputs:** Grad-CAM visualizations, overlays, publication-quality 300 DPI comparisons (PNG & SVG), raw arrays, and metadata JSON.
* **Python Libraries:** `torch`, `matplotlib`, `numpy`.
* **Folder Location:** `src/models/gradcam.py` and `artifacts/gradcam/`

---

## **6. Data Flow**

### **Deep-Dive Data Transformations**

1. **Remote Telemetry Ingestion:** Multi-extension FITS matrices are processed by applying target pixel masks and extracting integrated flux timelines (containing time and PDCSAP Flux).  
2. **Systematics Rectification:** Raw timelines undergo asymmetric sigma clipping ($+5\sigma$ / $-20\sigma$) to clean noise, followed by biweight detrending to remove slow stellar rotational signals.  
3. **Periodicity Search & Resolution Phase-Folding:** Box Least Squares (BLS) estimates the transit period and epoch ($T_0$) to fold and center the flux timeline at Phase $0.0$.  
4. **Feature Binning:** The folded timeline is binned into two resolution profiles: a Global View (2001 bins) capturing the entire orbital phase and a Local View (201 bins) isolating the transit event. Host-star properties are loaded from primary FITS headers. Output is serialized into `final_dataset.npz`.  
5. **Dual-Branch CNN Feature Extraction:** The Global View passes through 1D Convolutional layers to capture long-duration variability, while the Local View passes through parallel 1D Convolutional layers to extract transit slopes.  
6. **Penultimate Representation Extraction:** Deep spatial features are mapped through the first linear blocks of the classifier head to extract a unified 128-dimensional representation vector.  
7. **Feature Fusion:** The 128-dimensional spatial representation is concatenated with the 3 raw host-star properties ($T_{eff}$, $R_*$, $\log g$), generating a 131-dimensional hybrid vector.  
8. **Downstream GBDT Classification:** The 131-dimensional vector feeds the LightGBM classifier to produce the final planet candidate probability.
9. **Explainability Attribution mapping (GBDT):** The LightGBM classifier weights are evaluated using SHAP game-theoretic Shapley values, calculating direct feature attributions for all 131 parameters and plotting summaries.
10. **Explainability Spatial mapping (CNN):** The CNN encoder's features are evaluated via Grad-CAM, hooking into the last Conv1D layer of the Global and Local branches to map gradients back to binned input fluxes. This generates spatial heatmaps, overlays, and comparison grids in PNG and SVG formats.

---

## **7. Folder Structure**

The project directory structure is organized as follows:

* `StarSight/`: Root repository directory.  
* `docs/`: Technical audit reports, system architecture specs, and documentation.  
* `data/raw/`: Kepler raw FITS files downloaded from the MAST archive.  
* `data/processed/`: Cleaned target NPZs, binned feature arrays, and the unified `final_dataset.npz`.  
* `notebooks/`: Orchestration notebooks (01 to 05) designed for local or Colab execution.  
* `src/`: Core Python source code:
  * `config.py`: Centralized configuration variables (directories, hyperparameters, compute devices).
  * `experiment/experiment.py`: Lifecycle manager orchestrating checkpoints, snapshots, evaluations, and hybrid classifier training.
  * `models/`: Neural network architecture, datasets, metrics, and training loops.
  * `utils/`: Environmental checks, repository bootstrappers, and random seeds utilities.
  * `visualization/`: Matplotlib plotting scripts for training history and evaluations.
* `results/`: Stores transit metrics (`transit_summary.csv`), detrending plots, and periodic folded curves.  
* `artifacts/`: Houses model weights checkpoints (`best_model.pt`, `final_model.pt`, `cnn_encoder.pt`, `lightgbm_model.txt`), feature embeddings (`feature_embeddings.npz`), configuration snapshots, diagnostic loss/ROC curves, comparison metrics/plots, SHAP explainability visualizations/arrays (`explainability/`), Grad-CAM spatial visualizations/numpy arrays/validation logs (`gradcam/` & `validation/`), and logs.

---

## **8. Technology Stack**

The core tools utilized in the StarSight implementation:

| Component | Tech Suite | Justification |
| :--- | :--- | :--- |
| **Language & IDE** | Python 3.12 (Colab standard) / 3.13 (local), Jupyter | Standard tools for astronomical scripting, prototyping, and execution. |
| **Deep Learning** | PyTorch (`torch`) | Enables modular 1D-CNN architectures, late feature fusion, and penultimate embedding extraction. |
| **Ensemble Learning** | LightGBM (`lightgbm`) | Lightweight gradient boosting classifier optimized for high-dimensional tabular inputs and extreme class imbalance. |
| **Interpretability (Tabular)** | SHAP (`shap`) | SHapley Additive exPlanations module providing global and local attribution mappings. |
| **Interpretability (CNN)** | Grad-CAM (`torch` hooks) | Hook-based gradient-weighted class activation mapping for Global (2001) and Local (201) CNN branches. |
| **Core Utilities** | Lightkurve, Astropy, NumPy, Pandas, SciPy | Standard toolset for parsing FITS telemetry, running BLS sweeps, and processing arrays. |
| **Visualization** | Matplotlib | Generates diagnostic curves, confusion matrices, model comparison plots, SHAP diagrams, and Grad-CAM overlays. |
| **Version Control** | Git, GitHub | Source code tracking and remote synchronization. |

---

## **9. Model Evaluation**

* **Evaluation Metrics:** The system evaluates standalone CNN and hybrid CNN+LightGBM classifications using Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrices, and Training/Inference execution times.
* **Validation Strategy:** Enforces a Group-stratified train/validation/test split based on host-star target IDs (using a fixed seed 42) to eliminate temporal leakage.
* **Outputs Generated:** Performance is saved inside the `artifacts/` folder as `evaluation_metrics.csv`, `model_comparison_metrics.csv`, model comparison curves (`model_comparison_plots.png`), SHAP explainability files (`explainability/`), Grad-CAM files (`gradcam/`), and validation assertions report (`validation/gradcam_validation.json`).

---

## **10. Expected Output**

* **Input:** Raw FITS Light Curve.  
* **Output:** Planet Candidate Probability, Transit Period, Transit Duration, Epoch ($T_0$), Model Comparison Metrics, and SHAP Interpretability attributions.

---

## **11. Infrastructure Costs**

* **Compute Environment:** StarSight runs completely on free Google Colab container runtimes (leveraging T4 GPUs) or local consumer hardware (utilizing CPU or Apple Silicon MPS GPU acceleration). No high-cost cloud computing instances or distributed databases are required.

---

## **12. Future Enhancements**

The following features represent planned extensions and are not included in the current core implementation:

* **Interactive Frontend Dashboard:** An interactive web dashboard utilizing Streamlit or Plotly for real-time target visualization and prediction vetting.
* **TESS Multi-Mission Scaling:** Extending telemetry ingestion routines to process high-noise, short-baseline TESS (Transiting Exoplanet Survey Satellite) profiles.
* **Signal Denoising:** Advanced filtering using Discrete Wavelet Transforms (DWT) to separate stellar flares from transits.
* **Inference API:** A lightweight REST API for serving classification predictions in real time.