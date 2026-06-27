import sys
import torch
from pathlib import Path
import src.config as config

def detect_colab() -> bool:
    """
    Detects if the code is executing in a Google Colab environment.
    
    Returns:
        bool: True if in Colab, False otherwise.
    """
    return 'google.colab' in sys.modules

def mount_drive_if_needed() -> None:
    """
    Mounts Google Drive under /content/drive if executing in Colab.
    """
    if detect_colab():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully.")
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")
    else:
        print("Not running in Google Colab. Skipping Drive mount.")

def print_environment_summary() -> None:
    """
    Prints a formatted summary of the active environment paths and device properties.
    """
    is_colab = detect_colab()
    env_str = "Google Colab (Cloud)" if is_colab else "Local Machine (Desktop)"
    
    print("=" * 60)
    print("                 STARSIGHT ENVIRONMENT SUMMARY")
    print("=" * 60)
    print(f"Execution Environment  : {env_str}")
    print(f"Active Compute Device  : {config.DEVICE}")
    print(f"Project Base Directory : {config.BASE_DIR.resolve()}")
    print(f"Raw Data Directory     : {config.RAW_DIR.resolve()}")
    print(f"Processed Directory    : {config.PROCESSED_DIR.resolve()}")
    print(f"Results Directory      : {config.RESULTS_DIR.resolve()}")
    print(f"Artifact Directory     : {config.ARTIFACTS_DIR.resolve()}")
    print(f"Predictions Directory  : {config.PREDICTIONS_DIR.resolve()}")
    print(f"Config Snapshot Dir    : {config.CONFIGS_DIR.resolve()}")
    print(f"Summaries Directory    : {config.SUMMARIES_DIR.resolve()}")
    print("=" * 60)
