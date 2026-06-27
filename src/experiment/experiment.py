import json
import time
import datetime
import logging
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple

# Import decoupled components
from src.utils.reproducibility import set_seed
from src.models.datasets import create_data_loaders
from src.models.metrics import calculate_metrics
from src.visualization.plots import plot_diagnostic_curves, plot_roc_and_confusion_matrix

class Experiment:
    """
    Manages and coordinates the complete lifecycle of a model training experiment.
    Decouples execution modules (trainer, metrics, plots) from database I/O, 
    and captures reproducibility telemetry.
    """
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.datetime.now().isoformat()
        self.experiment_id = f"exp_{int(time.time())}"
        self.logger = self._setup_logging()
        self.git_hash = self._get_git_commit_hash()
        
    def _setup_logging(self) -> logging.Logger:
        """Sets up the file and console logging handlers under logs/."""
        self.config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = self.config.LOGS_DIR / "training.log"
        
        logger = logging.getLogger("StarSight.Experiment")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Also print to stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(stream_handler)
        
        return logger

    def _get_git_commit_hash(self) -> str:
        """Attempts to dynamically read the current git commit hash."""
        try:
            cmd = ["git", "rev-parse", "HEAD"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return result.stdout.strip()
        except Exception:
            self.logger.warning("Could not retrieve Git commit hash. Defaulting to 'n/a'.")
            return "n/a"

    def save_config_snapshot(self) -> None:
        """Saves a JSON snapshot of the training parameters to configs/config.json."""
        self.config.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        
        snapshot = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "git_commit_hash": self.git_hash,
            "learning_rate": self.config.LEARNING_RATE,
            "epochs": self.config.MAX_EPOCHS,
            "random_seed": self.config.RANDOM_SEED,
            "early_stopping_patience": self.config.EARLY_STOPPING_PATIENCE,
            "device": str(self.config.DEVICE),
            "dev_mode": self.config.DEV_MODE,
            "dataset_path": str((self.config.PROCESSED_DIR / "final_dataset.npz").resolve()),
            "optimizer": "Adam",
            "loss_criterion": "BCEWithLogitsLoss"
        }
        
        snapshot_path = self.config.CONFIGS_DIR / "config.json"
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=4)
        self.logger.info(f"Configuration snapshot saved to {snapshot_path.resolve()}")

    def run_cnn_experiment(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, Any]:
        """
        Orchestrates dataset creation, PyTorch model training, checkpoints saving,
        inference, metrics computation, plotting, and exports run diagnostics.
        """
        self.logger.info(f"Starting CNN Experiment: {self.experiment_id}")
        self.save_config_snapshot()
        
        # 1. Prepare Datasets & DataLoaders
        data_path = self.config.PROCESSED_DIR / "final_dataset.npz"
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(data_path, self.config)
        
        # Read dataset statistics
        train_size = len(train_dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        total_dataset_size = train_size + val_size + test_size
        
        self.logger.info(f"Dataset split counts: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # 2. Run Training Loop epoch-by-epoch
        start_time = time.time()
        
        from src.models.trainer import train_epoch, evaluate_epoch
        
        history = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(1, self.config.MAX_EPOCHS + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, self.config.DEVICE)
            val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, self.config.DEVICE)
            
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
            history.append(epoch_record)
            
            log_msg = (f"Epoch {epoch:02d}/{self.config.MAX_EPOCHS:02d} | "
                       f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
                       f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            self.logger.info(log_msg)
            
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_path = self.config.MODELS_DIR / "best_model.pt"
                model.save_checkpoint(best_model_path)
                self.logger.info(f"Validation loss improved. Saved best model to {best_model_path.name}")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
        duration = time.time() - start_time
        self.logger.info(f"Training loop execution finished in {duration:.2f} seconds.")
        
        # Save final model state using helper save_checkpoint
        final_model_path = self.config.MODELS_DIR / "final_model.pt"
        model.save_checkpoint(final_model_path)
        self.logger.info(f"Saved final model states to {final_model_path.name}")
        
        # Save history CSV file
        self.config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        df_history = pd.DataFrame(history)
        history_csv_path = self.config.METRICS_DIR / "history.csv"
        df_history.to_csv(history_csv_path, index=False)
        self.logger.info(f"Saved epoch logs spreadsheet to {history_csv_path.name}")
        
        # Determine best validation epoch index
        val_losses = [epoch_log["val_loss"] for epoch_log in history]
        best_epoch = int(np.argmin(val_losses)) + 1 if len(val_losses) > 0 else 1
        
        # 3. Test Set Inference & Decoupled Metrics Calculation
        model.load_checkpoint(best_model_path, device=self.config.DEVICE)
        model.eval()
        
        y_true_list = []
        y_pred_probs_list = []
        
        with torch.no_grad():
            for x_global, x_local, x_stellar, y_true in test_loader:
                x_global = x_global.to(self.config.DEVICE)
                x_local = x_local.to(self.config.DEVICE)
                x_stellar = x_stellar.to(self.config.DEVICE)
                
                logits = model(x_global, x_local, x_stellar)
                probs = torch.sigmoid(logits)
                
                y_true_list.extend(y_true.cpu().numpy().flatten())
                y_pred_probs_list.extend(probs.cpu().numpy().flatten())
                
        y_true = np.array(y_true_list)
        y_pred_probs = np.array(y_pred_probs_list)
        
        # Save prediction outputs
        self.config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        predictions_path = self.config.PREDICTIONS_DIR / "test_predictions.npz"
        np.savez_compressed(predictions_path, y_true=y_true, y_pred_probs=y_pred_probs)
        self.logger.info(f"Test set predictions saved to {predictions_path.resolve()}")
        
        # Decoupled metrics calculations (returns dictionary)
        final_metrics = calculate_metrics(y_true, y_pred_probs)
        
        # Save final metrics summary to CSV
        df_metrics = pd.DataFrame([final_metrics])
        metrics_csv_path = self.config.METRICS_DIR / "evaluation_metrics.csv"
        df_metrics.to_csv(metrics_csv_path, index=False)
        self.logger.info(f"Saved test evaluation metrics to {metrics_csv_path.name}")
        
        # 4. Decoupled Plotting Visualization
        plot_diagnostic_curves(history, self.config.FIGURES_DIR)
        plot_roc_and_confusion_matrix(y_true, y_pred_probs, self.config.FIGURES_DIR)
        
        # 5. Export Experiment Summary JSON
        summary = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "device": str(self.config.DEVICE),
            "dataset_size": total_dataset_size,
            "split_sizes": {
                "train": train_size,
                "validation": val_size,
                "test": test_size
            },
            "best_epoch": best_epoch,
            "final_metrics": final_metrics,
            "saved_model_paths": {
                "best_model": str((self.config.MODELS_DIR / "best_model.pt").resolve()),
                "final_model": str((self.config.MODELS_DIR / "final_model.pt").resolve())
            },
            "figure_paths": [
                str((self.config.FIGURES_DIR / "loss_accuracy_curves.png").resolve()),
                str((self.config.FIGURES_DIR / "roc_confusion_matrix.png").resolve())
            ],
            "training_duration_seconds": float(duration)
        }
        
        summary_path = self.config.METRICS_DIR / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"Experiment summary telemetry exported to {summary_path.resolve()}")
        
        return summary
