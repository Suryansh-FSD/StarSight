# **StarSight: AI-Enabled Detection of Exoplanets**

StarSight is an end-to-end scientific pipeline for detecting periodic transit signals of exoplanets from raw astronomical light curves using deep learning.

---

## **1. Core Pipeline Stages**

The pipeline is organized into five sequential steps, each driven by a structured notebook in `notebooks/` and backed by modular utility classes in `src/`:

1. **01_data_acquisition.ipynb:** Downloads raw FITS observation light curves for Kepler targets from the STScI MAST archive.
2. **02_preprocessing_detrending.ipynb:** Cleans the raw PDCSAP flux. Applies NaN removal, asymmetric outlier sigma clipping ($+5\sigma$ / $-20\sigma$), normalization, and biweight filtering (0.5 day window) to remove low-frequency stellar rotational variability.
3. **03_transit_search.ipynb:** Sweeps the frequency space using Box Least Squares (BLS) to calculate periodograms and identify candidate transit periods, epochs ($T_0$), depths, and durations.
4. **04_astronet_features.ipynb:** Phase-folds light curves centered at phase $0.0$ and extracts stellar parameters ($T_{eff}$, $R_*$, $\log g$) from FITS primary headers. Generates standardized binned arrays: a Global View (2001 bins) and a zoomed-in Local View (201 bins). Serializes final datasets to `final_dataset.npz`.
5. **05_cnn_training.ipynb:** Trains a late-fusion Dual-Branch CNN model (AstroNet format) on the engineered datasets using early stopping, extracts penultimate 128-dimensional spatial representations, concatenates them with host star physical attributes, and trains a downstream hybrid LightGBM classifier. Runs SHAP explainability analyses (global feature importance, summary beeswarm, waterfall, and force plots) to automatically interpret exoplanet classification outputs. Runs on PyTorch compute accelerators (MPS/CUDA/CPU).

---

## **2. Directory Structure**

* `src/`: Core Python implementations:
  * `config.py`: Centralized training hyperparameters, compute device cascade, and path definitions.
  * `experiment/experiment.py`: Orchestrator managing training configurations, snapshots, and evaluations.
  * `models/`: PyTorch neural networks architectures, dataset loaders, metrics, and training loops.
  * `utils/`: Environmental checks, repository bootstrappers, and deterministic random seeds.
  * `visualization/`: Matplotlib plotting scripts for training curves.
* `notebooks/`: Prototyping and orchestration notebooks (01 to 05).
* `data/`: Ingested FITS light curves and preprocessed NumPy `.npz` archives.
* `results/`: Detrending diagnostics, folded light curves, and the central `transit_summary.csv` sheet.
* `artifacts/`: Houses model checkpoints (`best_model.pt`, `final_model.pt`, `cnn_encoder.pt`, `lightgbm_model.txt`), feature embeddings (`feature_embeddings.npz`), performance curves, metrics logs, SHAP plots/arrays (`explainability/`), and run telemetry.
* `docs/`: Technical specifications and audits.

---

## **3. Technology Stack**

* **Language & Runtime:** Python 3.12 (Colab standard) / 3.13 (local), Jupyter
* **Deep Learning Framework:** PyTorch (`torch`)
* **Core Libraries:** Lightkurve, Astropy, NumPy, Pandas, SciPy, Matplotlib, LightGBM, SHAP

---

## **4. Future Enhancements**

The following features represent planned production extensions and are not part of the current core implementation:
* **Interactive Dashboard:** Streamlit or Plotly web application interface for real-time visual vetting.
* **TESS Multi-Mission Scaling:** Telemetry ingestion scaling to process TESS (Transiting Exoplanet Survey Satellite) profiles.
* **Denoising Filters:** Signal filtering using Discrete Wavelet Transforms (DWT) to separate stellar flares from planetary transits.
* **Inference API:** A lightweight REST API serving exoplanet classification predictions in real time.
