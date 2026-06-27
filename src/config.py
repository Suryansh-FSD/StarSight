import sys
from pathlib import Path
import torch

# Part 1: Automatic Environment Detection
IS_COLAB = 'google.colab' in sys.modules

# Part 2 & 3: Resolve Persistent Project Root and Base Directory
if IS_COLAB:
    # Auto-mount Drive to access permanent cloud storage
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as e:
        print(f"Warning: Could not automatically mount Google Drive in config: {e}")
    BASE_DIR = Path('/content/drive/MyDrive/StarSight')
else:
    # Local path resolution (repository root)
    notebook_dir = Path.cwd()
    BASE_DIR = notebook_dir.parent if notebook_dir.name == "notebooks" else notebook_dir

# Execution Modes:
# True: Development Mode (runs on a small dataset verification)
# False: Training Mode (runs on the full dataset)
DEV_MODE = True

# Directory Paths (Separating code from persistent data in Colab)
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
CONFIGS_DIR = ARTIFACTS_DIR / "configs"
SUMMARIES_DIR = ARTIFACTS_DIR / "summaries"

# Part 4: Automatically create directories if missing
for folder in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR, PREDICTIONS_DIR, CONFIGS_DIR, SUMMARIES_DIR]:
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

GLOBAL_BINS = 2001
LOCAL_BINS = 201
