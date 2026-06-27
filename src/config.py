import sys
from pathlib import Path
import torch

# Detect environment and resolve base directory
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = Path('/content/drive/MyDrive/StarSight')
else:
    # Local path resolution
    notebook_dir = Path.cwd()
    BASE_DIR = notebook_dir.parent if notebook_dir.name == "notebooks" else notebook_dir

# Execution Modes:
# True: Development Mode (runs on a small dataset verification)
# False: Training Mode (runs on the full dataset)
DEV_MODE = True

# Directory Paths
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

# Centralized Artifact Directories
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "logs"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# Ensure all directories exist programmatically
for folder in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
RANDOM_SEED = 42
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# Compute Device Detection Cascade: CUDA -> Apple MPS -> CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
