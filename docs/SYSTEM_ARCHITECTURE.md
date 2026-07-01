# **StarSight**

## ***AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves***

**Document Reference:** ISRO-BAH-2026-EXO-ADS-ARCH-001  
**Classification:** Technical Design Document  
**Target Event:** Bharatiya Antariksh Hackathon 2026  
**System Name:** StarSight (Exoplanet Automated Detection System)

---

## **1. Executive Summary**

StarSight is an AI-powered system for detecting exoplanets from noisy astronomical light curves. The system downloads Kepler space telescope observations, preprocesses the raw time-series, detects potential transit events using Box Least Squares (BLS), generates standardized Global and Local Views, and classifies candidates using a custom late-fusion Dual-Branch Convolutional Neural Network (CNN) model (AstroNet format). The architecture is designed for rapid local and cloud container execution, scientific correctness, and verification during the Bharatiya Antariksh Hackathon 2026.

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
* **Integrated Late-Fusion Architecture:** Deploy a multi-modal network combining deep spatial feature extraction on light curves with tabular host-star physical attributes.

### **Scientific Objectives**

* **Stellar Context Grounding:** Incorporate host-star physical properties (radius $R_*$, effective temperature $T_{eff}$, and surface gravity $\log g$) directly into the final classification head to ground predictions in real stellar physics.  
* **Systematics Rectification:** Process raw light curves by applying NaN removal, asymmetric outlier sigma clipping, biweight detrending, and normalization to isolate the transit signal cleanly.  
* **Strict Reproducibility:** Ensure all random operations (data splits, model weights initialization, batch shuffling) are deterministic and governed by a centralized seed.

### **Hackathon & Deployment Scope**

* **Local & Cloud Compatibility:** Ensure the entire codebase runs identically on both a local development environment (leveraging Apple Silicon GPU/MPS) and Google Colab (utilizing CUDA GPU acceleration with Google Drive persistent storage).

---

## **4. System Workflow**

The StarSight end-to-end data processing workflow is organized into six sequential phases:  
**NASA Kepler Archive → Download FITS Files → PDCSAP Flux Extraction → Cleaning (NaN Removal, Asymmetric Sigma Clipping, Normalization, Biweight Detrending) → Box Least Squares (BLS) Periodogram Sweep → Phase Folding → AstroNet Global & Local Views Generation → Dual-Branch CNN Classification (Global + Local branches + Stellar parameters Late Fusion) → Planet Candidate Probability Output.**

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
I --> J[Late Feature Fusion + Stellar Parameters]
J --> K[Dense Classifier Head]
K --> L[Candidate Probability Output]
```

---

## **5. Software Architecture**

The granular breakdown of each software module:

### **Module 1: Telemetry Data Ingestion (data_acquisition)**
* **Purpose:** Stream raw multi-dimensional satellite telemetry files and isolate uncorrected target light curves.  
* **Responsibilities:** Parse multi-extension FITS files; pull spatial headers; compute integrated flux counts.  
* **Inputs:** Raw space archive `.fits` target pixel files.  
* **Outputs:** DataFrames containing Time ($t$) and PDCSAP Flux.  
* **Python Libraries:** `astropy.io.fits`, `lightkurve`, `numpy`, `pandas`.  
* **Folder Location:** `data/raw/`  

### **Module 2: Science Data Partitioning Engine (datasets)**
* **Purpose:** Partition, structure, and serve data inputs under strict scientific cross-validation rules.  
* **Responsibilities:** Enforce group-stratified splitting by unique target host star identifier to eliminate data leakage.  
* **Inputs:** Time-series binned arrays; tabular host-star physical properties.  
* **Outputs:** Train (9 samples), Validation (3 samples), and Test (4 samples) PyTorch DataLoaders.  
* **Python Libraries:** `pandas`, `scikit-learn` (`train_test_split`).  
* **Folder Location:** `data/processed/`  

### **Module 3: Systematics Rectification Core (preprocessing)**
* **Purpose:** Cleanse signal data for downstream classification while preserving physical transit geometric shapes.  
* **Responsibilities:** Apply asymmetric outlier clipping ($+5\sigma$ to remove positive cosmic ray spikes, protecting $-20\sigma$ transit dips), perform windowed biweight detrending (0.5 day window) to isolate stellar rotation, and normalize flux.  
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

### **Module 6: Trimodal Deep Representation Engine & Classifier (astronet)**
* **Purpose:** Extract spatial feature embeddings from multi-resolution phase-folded views and output exoplanet candidate probabilities.  
* **Responsibilities:** Process Global Views via 1D Convolutional layers, process Local Views via parallel 1D Convolutional layers, process tabular stellar metadata via fully connected layers, concatenate representations, and map to a final classification output.  
* **Inputs:** Global and Local binned arrays, tabular stellar parameters.  
* **Outputs:** Continuous planet candidate probability in the range $[0, 1]$.  
* **Python Libraries:** `torch`.  
* **Folder Location:** `src/models/`  

---

## **6. Data Flow**

### **Deep-Dive Data Transformations**

1. **Remote Telemetry Ingestion:** Multi-extension FITS matrices are processed by applying target pixel masks and extracting integrated flux timelines (containing time and PDCSAP Flux).  
2. **Systematics Rectification:** Raw timelines undergo asymmetric sigma clipping ($+5\sigma$ / $-20\sigma$) to clean noise, followed by biweight detrending to remove slow stellar rotational signals.  
3. **Periodicity Search & Resolution Phase-Folding:** Box Least Squares (BLS) estimates the transit period and epoch ($T_0$) to fold and center the flux timeline at Phase $0.0$.  
4. **Feature Binning:** The folded timeline is binned into two resolution profiles: a Global View (2001 bins) capturing the entire orbital phase and a Local View (201 bins) isolating the transit event. Host-star properties are loaded from primary FITS headers. Output is serialized into `final_dataset.npz`.  
5. **Dual-Branch CNN Feature Extraction:** The Global View passes through 1D Convolutional layers to capture long-duration variability, while the Local View passes through parallel 1D Convolutional layers to extract transit slopes. Tabular stellar properties pass through a linear layer.  
6. **Feature Fusion & Classification:** Deep CNN feature outputs are fused with the tabular stellar properties, producing a combined 67-element representation vector. A fully connected dense layer maps this vector to a single raw logit, which is activated via a Sigmoid function to output the final planet candidate probability.

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
  * `experiment/experiment.py`: Lifecycle manager orchestrating checkpoints, snapshots, and evaluations.
  * `models/`: Neural network architecture, datasets, metrics, and training loops.
  * `utils/`: Environmental checks, repository bootstrappers, and random seeds utilities.
  * `visualization/`: Matplotlib plotting scripts for training history and evaluations.
* `results/`: Stores transit metrics (`transit_summary.csv`), detrending plots, and periodic folded curves.  
* `artifacts/`: Houses model weights checkpoints (`best_model.pt`, `final_model.pt`), configuration snapshots, diagnostic loss/ROC curves, metrics spreadsheets, and logs.

---

## **8. Technology Stack**

The core tools utilized in the StarSight implementation:

| Component | Tech Suite | Justification |
| :--- | :--- | :--- |
| **Language & IDE** | Python 3.12 (Colab standard) / 3.13 (local), Jupyter | Standard tools for astronomical scripting, prototyping, and execution. |
| **Deep Learning** | PyTorch (`torch`) | Enables modular 1D-CNN architectures, late feature fusion, and late activation. |
| **Core Utilities** | Lightkurve, Astropy, NumPy, Pandas, SciPy | Standard toolset for parsing FITS telemetry, running BLS sweeps, and processing arrays. |
| **Visualization** | Matplotlib | Generates diagnostic curves, confusion matrices, and ROC plots. |
| **Version Control** | Git, GitHub | Source code tracking and remote synchronization. |

---

## **9. Model Evaluation**

* **Evaluation Metrics:** The system evaluates classifications using Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrices.
* **Validation Strategy:** Enforces a Group-stratified train/validation/test split based on host-star target IDs (using a fixed seed 42) to eliminate temporal leakage.
* **Outputs Generated:** Model performance is saved inside the `artifacts/` folder as `history.csv`, `evaluation_metrics.csv`, and associated diagnostic figures.

---

## **10. Expected Output**

* **Input:** Raw FITS Light Curve.  
* **Output:** Planet Candidate Probability, Transit Period, Transit Duration, Epoch ($T_0$).

---

## **11. Infrastructure Costs**

* **Compute Environment:** StarSight runs completely on free Google Colab container runtimes (leveraging T4 GPUs) or local consumer hardware (utilizing CPU or Apple Silicon MPS GPU acceleration). No high-cost cloud computing instances or distributed databases are required.

---

## **12. Future Enhancements**

The following features represent planned extensions and are not included in the current core implementation:

* **Interactive Frontend Dashboard:** An interactive web dashboard utilizing Streamlit or Plotly for real-time target visualization and prediction vetting.
* **Integrated Explainability Engine:** Explainable AI (XAI) mapping using SHAP (SHapley Additive exPlanations) or Grad-CAM to plot feature attributions and isolate transit boundaries.
* **Gradient-Boosting Classifier Head:** Integration of a leaf-wise LightGBM classifier head downstream of the CNN feature extractor to optimize decision boundaries on highly imbalanced datasets.
* **TESS Multi-Mission Scaling:** Extending telemetry ingestion routines to process high-noise, short-baseline TESS (Transiting Exoplanet Survey Satellite) profiles.
* **Signal Denoising:** Advanced filtering using Discrete Wavelet Transforms (DWT) to separate stellar flares from transits.
* **Inference API:** A lightweight REST API for serving classification predictions in real time.