# **StarSight**

## ***AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves***

**Document Reference:** ISRO-BAH-2026-EXO-ADS-ARCH-001  
**Classification:** Technical Design Document  
**Target Event:** Bharatiya Antariksh Hackathon 2026  
**System Name:** StarSight (Exoplanet Automated Detection System)

## **1\. Executive Summary**

StarSight is an AI-powered system for detecting exoplanets from noisy astronomical light curves. The system downloads Kepler observations, preprocesses the data, detects potential transit events using Box Least Squares (BLS), generates Global and Local Views, classifies candidates using a hybrid Dual-Branch CNN and LightGBM model, explains predictions using SHAP, and presents results through an interactive Streamlit dashboard. The architecture is designed for rapid implementation, scientific correctness, and demonstration during the Bharatiya Antariksh Hackathon 2026\.

## **2\. Problem Statement**

Automated exoplanet detection pipelines operating on space-based photometric datasets face persistent data challenges that degrade naive machine learning configurations:

1. **Extreme Class Imbalance:** Labeled Threshold Crossing Event (TCE) datasets exhibit massive structural skews. Non-planetary noise and astrophysical false alarms outnumber valid planet candidates by ratios frequently exceeding 136:1, causing standard classifiers to stall or default to majority-class predictions.  
2. **Multi-Faceted Noise Profiles:** Space telescope telemetry is heavily obscured by high-frequency instrumental artifacts (pointing jitters, thermal drifts, aperture spills), localized cosmic ray pixel spikes, and natural stellar variability (pulsations, flares, starspots).  
3. **Astrophysical Confounding Modulations:** Grazing or detached Eclipsing Binaries (EBs) and background star light dilutions in crowded fields yield light curve geometries that closely mimic the shallow, periodic flux drops of genuine transiting exoplanets.  
4. **Box Least Squares (BLS):** This method is used to estimate the most likely orbital period and transit epoch before phase folding.

## **3\. Objectives**

### **Technical Objectives**

* **Optimized Multi-Modal Pipeline:** Build an integrated data parsing and execution stream optimized for quick prototyping and low computational overhead.  
* **Leakage-Protected Partitioning:** Implement data storage structures that enforce strict target-star stratified splits to prevent temporal leakage across evaluation pools.  
* **Decoupled Classification Head:** Deploy a hybrid pipeline separating deep spatial feature extraction from non-linear boundary optimization to maximize precision-recall metrics.

### **Scientific Objectives**

* **Stellar Context Grounding:** Incorporate host-star physical properties (radius R⊂, effective temperature Teff, and surface gravity log g) alongside time-series feature maps to ground predictions in real stellar physics.  
* **Systematics Rectification:** The project uses the PDCSAP flux already provided by the official Kepler pipeline and performs additional preprocessing including NaN removal, sigma clipping, normalization, detrending, and outlier removal.  
* **High-Fidelity Vetting:** Generate transparent post-hoc interpretability mappings to isolate exact physical transit boundaries (ingress, egress, flat bottom).

### **Expected Impact**

StarSight transitions exoplanet identification from a manual, resource-intensive screening workflow into an automated, high-precision ground segment triage platform. By ensuring low false-alarm distributions and maintaining cross-mission generalizability, the framework demonstrates a clear, practical execution path suitable for backing ISRO's upcoming space science footprint.

## **4\. System Workflow**

The StarSight end-to-end data processing workflow is organized into eight logical, sequential phases:  
**NASA Kepler Archive → Download FITS Files → PDCSAP Flux → Cleaning (Remove NaNs, Sigma Clipping, Normalization, Detrending) → Box Least Squares (BLS) → Phase Folding → Generate Global View (2001 bins) → Generate Local View (201 bins) → Dual 1D CNN (Global Branch \+ Local Branch) → Feature Fusion (CNN Features \+ Stellar Parameters) → LightGBM → Prediction → SHAP Explainability → Streamlit Dashboard**

```
graph TD
A[NASA Kepler Archive] --> B[Download FITS Files] --> C[PDCSAP Flux] --> D[Cleaning] --> E[Box Least Squares] --> F[Phase Folding] --> G[Global View (2001)] --> H[Local View (201)] --> I[Dual 1D CNN] --> J[Feature Fusion] --> K[LightGBM] --> L[Prediction] --> M[SHAP] --> N[Streamlit Dashboard]
```

## **5\. Software Architecture**

The granular breakdown of each software module under hackathon design constraints:

### **Module 1: Telemetry Data Ingestion (data\_ingestion)**

* **Purpose:** Stream raw multi-dimensional satellite telemetry files and isolate uncorrected target light curves.  
* **Responsibilities:** Parse multi-extension FITS files; pull spatial headers; compute integrated flux counts.  
* **Inputs:** Raw space archive .fits or .fits.gz target pixel files; target coordinates.  
* **Outputs:** DataFrames containing Time (t) and Flux.  
* **Python Libraries:** astropy.io.fits, lightkurve, numpy, pandas.  
* **Folder Location:** data/raw/  
* **Challenges:** Large compressed FITS structures can cause memory saturation during parallel unpacking loops.  
* **Best Practices:** Stream rows lazily and execute vectorized aperture masking sums over pre-allocated arrays.

### **Module 2: Science Data Storage Engine (data\_storage)**

* **Purpose:** Partition, structure, and serve data inputs under strict scientific cross-validation rules.  
* **Responsibilities:** Maintain baseline class proportions; enforce group-stratified splitting by unique target host star identifier (KIC ID) to eliminate temporal data leakage.  
* **Inputs:** Time-series flux records; tabular ground-truth catalog records.  
* **Outputs:** Indexed dataset splits partitioned into Train (80%), Validation (10%), and Test (10%) arrays.  
* **Python Libraries:** pandas, scikit-learn (specifically StratifiedGroupKFold).  
* **Folder Location:** data/  
* **Challenges:** Concurrent random disk read workflows from thousands of small target files create data fetch bottlenecks.  
* **Best Practices:** Structure pre-processed matrices into consolidated NumPy arrays and CSV metadata indexed by coordinate sectors.

### **Module 3: Systematics Rectification Core (preprocessing)**

* **Purpose:** Cleanse signal data for downstream classification while preserving physical transit geometric shapes.  
* **Responsibilities:** The project uses the PDCSAP flux already provided by the official Kepler pipeline and performs additional preprocessing including NaN removal, sigma clipping, normalization, detrending, and outlier removal.  
* **Inputs:** FITS target pixel files and light curves.  
* **Outputs:** Preprocessed flux timelines.  
* **Python Libraries:** scipy.optimize, statsmodels, lightkurve.correctors.  
* **Folder Location:** data/processed/  
* **Challenges:** Aggressive detrending filters can accidentally smooth out low-amplitude ingress/egress profiles.  
* **Best Practices:** Deploy windowed median filtering iteratively and run quality flag filters to bypass corrupted observation segments.

### **Module 4: Periodicity & Folding Processor (feature\_engineering)**

* **Purpose:** Box Least Squares (BLS) is used to estimate the most likely orbital period and transit epoch before phase folding.  
* **Responsibilities:** Sweep frequency spaces for period (P) and epoch (T0); stack continuous timelines; bin profiles into high-density grids.  
* **Inputs:** Corrected PDCSAP flux DataFrames.  
* **Outputs:** Tuple of normalized arrays: Global View (2,001 bins) and Local View (201 bins).  
* **Python Libraries:** astropy.timeseries.BoxLeastSquares, numpy.  
* **Folder Location:** data/processed/  
* **Challenges:** Accurate period determination requires sufficient baseline coverage.  
* **Best Practices:** Restrict the periodogram search grid to physical exoplanet ranges and use coarse iterations followed by localized optimization steps.

### **Module 5: Trimodal Deep Representation Engine (deep\_learning)**

* **Purpose:** Extract structural latent feature embeddings from multi-resolution phase-folded views.  
* **Responsibilities:** Map long-duration variations via a Global 1D-CNN branch; isolate edge structures via a Local 1D-CNN branch.  
* **Inputs:** Global and Local binned arrays.  
* **Outputs:** Flat deep temporal feature vectors.  
* **Python Libraries:** torch, numpy.  
* **Folder Location:** models/  
* **Challenges:** Overparameterization risks causing the network to memorize mission noise artifacts.  
* **Best Practices:** Limit convolutional depth, implement dynamic dropout layers, and apply batch normalization across steps.

### **Module 6: Boosting Ensemble Triage Classifier (classification)**

* **Purpose:** Resolve optimal non-linear decision boundaries on extracted feature matrices.  
* **Responsibilities:** Concatenate deep CNN temporal feature vectors with tabular host-star attributes (Teff, radius, log g); execute gradient-boosting runs.  
* **Inputs:** Latent feature maps; tabular stellar parameters; target training labels.  
* **Outputs:** Explicit categorical triage tags (Planet Candidate vs. False Positive) paired with certainty scores.  
* **Python Libraries:** lightgbm, scikit-learn.  
* **Folder Location:** models/  
* **Challenges:** Broken end-to-end backpropagation prevents simultaneous network and tree updates.  
* **Best Practices:** Train the CNN backbone until convergence, extract and cache stable embeddings into RAM, and use them to optimize tree parameters.

### **Module 7: Interpretability Verification Core (explainability)**

* **Purpose:** Provide transparent, physically grounded evaluations of internal model decisions.  
* **Responsibilities:** Compute SHAP attribution weights across the ensemble layers to highlight dominant scientific features.  
* **Inputs:** Trained LightGBM ensemble states; combined evaluation feature matrices.  
* **Outputs:** Tabular feature importances and localized waterfall attribution charts.  
* **Python Libraries:** shap, matplotlib.  
* **Folder Location:** results/  
* **Challenges:** High dimensionality can increase background compute time for kernel SHAP estimations.  
* **Best Practices:** Deploy TreeSHAP configurations to leverage tree optimizations for rapid calculations.

### **Module 8: Unified Operations Workspace (visualization)**

* **Purpose:** Deliver a scannable dashboard interface for visual inspection and project evaluation.  
* **Responsibilities:** Render binned time-series arrays; overlay model classifications; track parameter plots.  
* **Inputs:** Triage classification flags; target parameter matrices; SHAP attribution vectors.  
* **Outputs:** Operational frontend web workspace layout.  
* **Python Libraries:** streamlit, plotly.  
* **Folder Location:** app/  
* **Challenges:** High-density time-series plots can exhaust client-side browser rendering limits.  
* **Best Practices:** Use Plotly WebGL variants and apply simple bucket downsampling rules to out-of-transit light curve sections.

## **6\. Data Flow**

### **Deep-Dive Data Transformations**

1. **Remote Telemetry Ingestion:** Multi-extension FITS matrices (e.g., shape \[N\_cadences, 8, 8\]) are processed by applying target pixel masks and subtracting ambient radiation. This spatial aggregation compresses the multi-dimensional image cubes into a raw 1D integrated timeline (shape \[N\_cadences, 2\] containing time and FluxSAP).  
2. **Systematics Rectification:** The project uses the PDCSAP flux already provided by the official Kepler pipeline and performs additional preprocessing including NaN removal, sigma clipping, normalization, detrending, and outlier removal.  
3. **Periodicity Search & Resolution Phase-Folding:** Box Least Squares (BLS) is used to estimate the most likely orbital period and transit epoch before phase folding. The framework collapses the timeline using modulo operations, generating two distinct binned arrays: a macro-scale Global View (shape \[2001, 1\]) and a close-up Local View (shape \[201, 1\]).  
4. **Trimodal Deep Representation Engine:** The binned arrays pass through parallel 1D-CNN operations. The Global branch compresses the \[2001, 1\] tensor into a \[32\] feature map, while the Local branch transforms the \[201, 1\] tensor into a \[32\] feature map. These outputs are concatenated into a \[64\] vector and appended with a \[3\] tabular row hosting stellar catalog values (Teff, radius, log g), producing a final multi-modal embedding of shape \[67\].  
5. **Ensemble Triage Classification:** The unified \[67\] feature vector feeds directly into a leaf-wise LightGBM tree pipeline. The algorithm applies non-linear splits to compute classification probabilities, returning a structured prediction array of shape \[2\] representing explicit binary probabilities for Planet Candidate (PC) or Astrophysical False Positive (AFP) designations.  
6. **Explainability Vetting:** The final classification run undergoes attribution tracing via TreeSHAP. The operation maps a feature attribution vector of shape \[67\], which links prediction changes directly back to physical variables (e.g., transit depth, stellar radius), exposing any unphysical noise correlations.  
7. **Scientific Workspace:** The raw arrays, binned profiles, and SHAP distributions are passed to the Streamlit layout engine, rendering scannable UI plots for the evaluation panel.

## **7\. Folder Structure**

The enterprise open-source repository configuration partitions directories according to clean hackathon execution guidelines:

* StarSight/: Root repository directory.  
* docs/: Stores the design review text, manuals, and presentation materials.  
* data/raw/: Unprocessed FITS telemetry downloaded from space telescope archives.  
* data/processed/: Systematics-corrected, binned, and split NumPy arrays and CSV metadata ready for execution.  
* notebooks/: Google Colab prototyping scripts for fast experimentation.  
* models/: Houses PyTorch 1D-CNN model column architectures and LightGBM model parameter configurations.  
* utils/: Houses shared mathematical scripts (e.g., phase folding, detrending filters) to reduce duplicate definitions.  
* app/: Contains the frontend Streamlit visualization dashboard files.  
* results/: Stores model classification confusion matrices, weights, and SHAP attribution plots.  
* README.md: Main repository roadmap and installation manual.  
* requirements.txt: Complete baseline library pinning configuration.

## **8\. Technology Stack**

The selected tools maximize engineering velocity and compute efficiency within the sandbox limitations of a free Google Colab runtime landscape:

| Component | Tech Suite | Justification |
| :---- | :---- | :---- |
| Language & IDE | Python 3.11, Google Colab | Optimized for rapid prototyping and cloud execution during the hackathon. |
| AI Frameworks | PyTorch, LightGBM | Enables flexible 1D-CNN architectures and efficient gradient boosting for imbalanced data. |
| Core Utilities | Lightkurve, Astropy, NumPy, Pandas | Standard toolset for high-performance astronomical data analysis and processing. |
| Visualization & XAI | Streamlit, Plotly, SHAP | Facilitates transparent model interpretation and interactive UI deployment. |
| Version Control | Git, GitHub | Ensures robust collaborative development and source management. |

## **9\. AI Model Design**

## **10\. Model Evaluation**

### **Evaluation Metrics**

The system is evaluated based on: Precision, Recall, F1 Score, ROC-AUC, PR-AUC, and the Confusion Matrix.

### **Validation Strategy**

We employ a Stratified Train / Validation / Test Split (80/10/10) using Group-aware splitting by target star to ensure no data leakage.

### **Baselines**

StarSight is compared against: Classical BLS and CNN-only architectures.

### **Final Model**

The production model is a Dual-input CNN integrated with a LightGBM classification head.

### **Strategic Component Deconstructions**

* **Why Use a Global View?** The Global View (2,001 bins) spans the entire phase-folded period, mapping out-of-transit variations and identifying secondary eclipses to immediately separate eclipsing binary systems from planetary transits.  
* **Why Use a Local View?** The Local View (201 bins) isolates the transit event context to capture precise ingress/egress slopes, distinguishing flat-bottom U-shaped planet profiles from grazing V-shaped binary profiles.  
* **Why Parallel 1D-CNN Branches?** Avoids forcing a single column to parse structural features across different physical regimes, letting the Global column focus on macroscopic variations and the Local column track delicate micro-features.  
* **Why LightGBM after CNN Feature Extraction?** Standard fully connected neural network layers struggle to find clean decision boundaries on highly imbalanced (136:1) distributions. Tree-based gradient-boosting architectures efficiently handle skewed labels to optimize precision-recall balance.

## **11\. Comparison with Existing Methods**

## 

| Method | Limitations | StarSight Improvement |
| :---- | :---- | :---- |
| Classical BLS | High false positive rate; missing non-box signals. | ML-based triage significantly reduces false positives. |
| CNN Only | Struggles with extreme class imbalance. | Boosting head optimizes decision boundaries for imbalanced labels. |
| Manual Vetting | Time-intensive; non-scalable for large surveys. | Fully automated ground segment software. |
| AstroNet | Limited interpretability; complex deployment. | Integrated SHAP explainability and Streamlit dashboard. |

## **12\. Innovation**

## **13\. Expected Output**

**Input:** Raw FITS Light Curve.  
**Output:** Planet Probability, Transit Period, Transit Duration, Confidence Score, SHAP Explanation, and Interactive Visualization.

1. **Hybrid Deep Learning \+ Gradient Boosting:** Decouples temporal spatial shape extraction from decision boundary scaling to optimize F1 triage metrics quickly on limited hardware.  
2. **Explainable AI using SHAP:** Bridges the "black box" gap by presenting clear mathematical proof of physical feature dependency, providing judges with transparent scientific accountability.  
3. **Interactive Streamlit Dashboard:** Delivers a highly responsive, scannable user experience in pure Python to let vetting panels inspect coordinates and classification charts instantly.

## **14\. Risk Analysis**

## 

| Risk Domain | Likelihood | Impact | Mitigation Strategy   |
| :---- | :---- | :---- | :---- |
| Class Imbalance | High | Critical | Deploy custom class weighting parameters and horizontal time-series reflections within LightGBM configurations. |
| Noisy Light Curves | High | High | Apply PDCSAP preprocessing paired with windowed median detrending filters and outlier removal. |
| Model Overfitting | Medium | High | Limit network capacity by minimizing layer depth and implementing strict multi-fold cross-validation drops. |
| False Positives | High | High | Incorporate flat tabular host-star physical properties to actively cross-reference transit geometries. |
| Limited Colab Resources | High | Medium | Use lazy array streaming and leverage vectorized NumPy operations to manage VRAM footprints. |

## **15\. Hackathon Scope**

### **Version 1 (Must Have — Complete Implementation Bounds)**

* Automated download routines for Kepler data files via Lightkurve.  
* Pre-processing data cleaning pipelines (NaN removal, sigma clipping, PDCSAP preprocessing).  
* Box Least Squares (BLS) periodogram execution loops.  
* Global and Local view generation frames.  
* Dual-branch PyTorch 1D-CNN feature backbone.  
* LightGBM classifier integration.  
* SHAP feature attribution mapping.  
* Streamlit user application interface.

### **Version 2 (Future Scope — Production Extensions)**

* Multi-mission data ingestion scaling to ingest high-noise, short-baseline TESS profiles.  
* Advanced signal filtering utilizing Discrete Wavelet Transforms (DWT).  
* Structural sequence modeling via Self-Attention Transformer experiments.  
* Multi-task objective function parameter constraints.  
* Orchestrated cloud container deployment configurations.  
* Active operational synergy tracking alongside ISRO heliophysics baselines (Aditya-L1 proxies).

## **16\. Implementation Roadmap**

* **Phase 1 (Days 1–2):** Dataset Ingestion primitives, systematics preprocessing filters, and initial BLS orbital sweeps. Deliverables: Validated data loading components.  
* **Phase 2 (Days 2–3):** Phase-folding processing loops and Global/Local binned resolution array generation. Deliverables: Consolidated NumPy arrays and CSV metadata.  
* **Phase 3 (Days 3–4):** PyTorch 1D-CNN feature column compilation, late feature concatenation, and LightGBM tree optimization. Deliverables: Trained model checkpoints.  
* **Phase 4 (Day 4):** SHAP explainability attributions, Streamlit visual app assembly, and final presentation refinement. Deliverables: Live interface deployment via secure local tunnel.

## **17\. References**

1. Agnes, C. K., Naveed, A., & Chacko, A. M. M. O. (2022). ExoSGAN and ExoACGAN: Exoplanet Detection using Adversarial Training Algorithms. *International Research Journal of Science and Technology*, 9(11).  
2. Ansdell, M., et al. (2018). ExoNet: A Multimodal Deep Learning Architecture for Exoplanet Vetting. *Astronomical Journal Letters*.  
3. Cuéllar, S., Granados, P., Fabregas, E., Curé, M., Vargas, H., Dormido-Canto, S., & Farias, G. (2022). Deep learning exoplanets detection by combining real and synthetic data. *PLOS ONE*, 17(5), e0268199.  
4. Karimi, R., Mousavi-Sadr, M., Haghighi, M. H. Z., & Tabatabaei, F. S. (2025). Machine Learning for Exoplanet Detection: A Comparative Analysis Using Kepler Data. *arXiv preprint*.  
5. Montalto, M., Borsato, L., Granata, V., Lacedelli, G., Malavolta, L., Manthopoulou, E. E., Nardiello, D., Nascimbeni, V., & Piotto, G. (2020). A search for transiting planets around FGKM dwarfs and subgiants in the TESS full frame images of the Southern ecliptic hemisphere. *Monthly Notices of the Royal Astronomical Society*, 498(2), 1726-1749.  
6. Salinas, H., Pichara, R., Brahm, R., Pérez-Galarce, F., & Mery, D. (2023). Distinguishing a planetary transit from false positives: a Transformer-based classification for planetary transit signals. *Monthly Notices of the Royal Astronomical Society*, 522(3), 3201-3211.  
7. Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Convolutional Neural Network for Identifying Transiting Planets in the Kepler Light Curve. *The Astronomical Journal*, 155(2), 94\.

## **18\. Appendix: Glossary of Domain Terms**

### **Astronomical Terminology**

* **Simple Aperture Photometry (SAP):** The uncorrected integrated flux calculated by summing pixel counts over time inside a target aperture mask.  
* **Pre-search Data Conditioning SAP (PDCSAP):** The project uses the PDCSAP flux already provided by the official Kepler pipeline and performs additional preprocessing including NaN removal, sigma clipping, normalization, detrending, and outlier removal.  
* **Threshold Crossing Event (TCE):** A periodic signal drop that crosses a designated statistical noise threshold.  
* **Limb Darkening:** The physical phenomenon where a star's central disk appears brighter than its edges, causing planetary transits to show characteristic curved profiles.

### **Machine Learning Terminology**

* **Data Leakage:** An evaluation flaw where information from the testing pool accidentally contaminates the training subset, artificially inflating validation accuracy.  
* **Late Fusion:** A multimodal architecture design where separate data channels are processed by independent columns before being combined in final layers.

## **19\. Changelog**

Updated document title and subtitle to professional engineering standards.  
Reconfigured all folder path references to the official project repository structure.  
Replaced custom CBV/SVD preprocessing details with PDCSAP-integrated pipeline steps.  
Standardized BLS descriptions across 'Problem Statement' and 'Feature Engineering' modules.  
Overhauled 'System Workflow' to reflect current multi-modal data processing stages.  
Added 'Model Evaluation' section covering metrics, validation strategy, and baselines.  
Added 'Expected Output' and 'Comparison with Existing Methods' sections for stakeholder review.  
Revised 'Tech Stack' to reflect prioritized hackathon toolsets and streamlined justifications.

* 