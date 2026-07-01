import json
import time
import datetime
import logging
import subprocess
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any

# Import decoupled components
from src.models.datasets import create_data_loaders

from src.models.metrics import calculate_metrics
from src.visualization.plots import plot_diagnostic_curves, plot_roc_and_confusion_matrix
from src.models.registry import get_model

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
        self.git_branch = self._get_git_branch()
        
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

    def _get_git_branch(self) -> str:
        """Attempts to dynamically read the current git branch name."""
        try:
            cmd = ["git", "branch", "--show-current"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            branch = result.stdout.strip()
            if not branch:
                cmd_fallback = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
                result_fallback = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                branch = result_fallback.stdout.strip()
            return branch
        except Exception:
            self.logger.warning("Could not retrieve Git branch name. Defaulting to 'n/a'.")
            return "n/a"

    def save_config_snapshot(self) -> None:
        """Saves a JSON snapshot of the training parameters to configs/config.json."""
        self.config.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        
        snapshot = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "git_commit_hash": self.git_hash,
            "git_branch": self.git_branch,
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

    def run_experiment(self, model_name: str, model_kwargs: Dict[str, Any], optimizer_cls, criterion) -> Dict[str, Any]:
        """
        Orchestrates dataset creation, model registry lookups, PyTorch model training, 
        checkpoints saving, inference, metrics calculation, plotting, and exports run diagnostics.
        """
        self.logger.info(f"Starting Experiment Run: {self.experiment_id}")
        self.save_config_snapshot()
        
        # Ensure reproducibility of weight initialization and shuffle patterns
        from src.utils.reproducibility import set_seed
        set_seed(self.config.RANDOM_SEED)
        
        # 1. Instantiate Model from Registry
        model = get_model(model_name, **model_kwargs)

        
        # 2. Prepare Datasets & DataLoaders
        data_path = self.config.PROCESSED_DIR / "final_dataset.npz"
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(data_path, self.config)
        
        # Read dataset statistics
        train_size = len(train_dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        total_dataset_size = train_size + val_size + test_size
        
        self.logger.info(f"Dataset split counts: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Capture model summary structure and print
        summary_text = model.get_summary_text(
            global_shape=(1, self.config.GLOBAL_BINS),
            local_shape=(1, self.config.LOCAL_BINS),
            stellar_shape=(1, 3),
            device=self.config.DEVICE
        )
        print(summary_text)
        
        # Save model summary to disk
        self.config.SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        model_summary_path = self.config.SUMMARIES_DIR / "model_summary.txt"
        with open(model_summary_path, "w") as f:
            f.write(summary_text)
        self.logger.info(f"Model architecture summary saved to {model_summary_path.resolve()}")
        
        # Move model to target compute device before optimization
        model.to(self.config.DEVICE)
        
        # 3. Instantiate Optimizer
        optimizer = optimizer_cls(model.parameters(), lr=self.config.LEARNING_RATE)

        
        # 4. Run Training Loop epoch-by-epoch
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
        
        # 5. Test Set Inference & Decoupled Metrics Calculation
        model.load_checkpoint(best_model_path, device=self.config.DEVICE)
        model.eval()
        
        # Measure CNN Inference Time
        cnn_start_inf = time.time()
        y_true_list = []
        y_pred_probs_list = []
        
        with torch.no_grad():
            for x_global, x_local, x_stellar, y_true_batch in test_loader:
                x_global = x_global.to(self.config.DEVICE)
                x_local = x_local.to(self.config.DEVICE)
                x_stellar = x_stellar.to(self.config.DEVICE)
                
                logits = model(x_global, x_local, x_stellar)
                probs = torch.sigmoid(logits)
                
                y_true_list.extend(y_true_batch.cpu().numpy().flatten())
                y_pred_probs_list.extend(probs.cpu().numpy().flatten())
                
        cnn_inference_time = time.time() - cnn_start_inf
        y_true = np.array(y_true_list)
        y_pred_probs = np.array(y_pred_probs_list)
        
        # Save standard CNN predictions
        self.config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        predictions_path = self.config.PREDICTIONS_DIR / "test_predictions.npz"
        np.savez_compressed(predictions_path, y_true=y_true, y_pred_probs=y_pred_probs)
        self.logger.info(f"Test set predictions saved to {predictions_path.resolve()}")
        
        # Compute CNN Metrics
        cnn_metrics = calculate_metrics(y_true, y_pred_probs)
        cnn_metrics["training_time"] = float(duration)
        cnn_metrics["inference_time"] = float(cnn_inference_time)
        
        # Save standard CNN checkpoint as cnn_encoder.pt
        cnn_encoder_path = self.config.MODELS_DIR / "cnn_encoder.pt"
        model.save_checkpoint(cnn_encoder_path)
        self.logger.info(f"Saved CNN encoder checkpoint to {cnn_encoder_path.name}")
        
        # --- Train and Evaluate Hybrid CNN + LightGBM ---
        self.logger.info("Extracting feature embeddings from the penultimate layer...")
        X_train, y_train = self._extract_embeddings_and_labels(model, train_loader)
        X_val, y_val = self._extract_embeddings_and_labels(model, val_loader)
        X_test, y_test = self._extract_embeddings_and_labels(model, test_loader)
        
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()
        
        # Save feature embeddings NPZ
        embeddings_path = self.config.PREDICTIONS_DIR / "feature_embeddings.npz"
        np.savez_compressed(
            embeddings_path,
            X_train_emb=X_train, y_train=y_train,
            X_val_emb=X_val, y_val=y_val,
            X_test_emb=X_test, y_test=y_test
        )
        self.logger.info(f"Feature embeddings saved to {embeddings_path.resolve()}")
        
        import lightgbm as lgb
        
        self.logger.info("Training LightGBM Classifier on top of extracted embeddings...")
        lgb_start_train = time.time()
        
        # Set small leaf/split parameters for tiny demo dataset
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            min_child_samples=2,
            random_state=self.config.RANDOM_SEED,
            n_jobs=1,
            verbosity=-1
        )
        
        lgb_model.fit(X_train, y_train)
        lgb_train_time = time.time() - lgb_start_train
        hybrid_train_time = duration + lgb_train_time
        
        # Save LightGBM Model text
        lgb_path = self.config.MODELS_DIR / "lightgbm_model.txt"
        lgb_model.booster_.save_model(str(lgb_path))
        self.logger.info(f"Saved LightGBM model config text to {lgb_path.name}")
        
        # Inference for hybrid model
        hybrid_start_inf = time.time()
        hybrid_pred_probs = lgb_model.predict_proba(X_test)[:, 1]
        hybrid_inference_total_time = (time.time() - hybrid_start_inf) + cnn_inference_time
        
        # Save predictions probabilities of the hybrid model
        hybrid_preds_path = self.config.PREDICTIONS_DIR / "test_predictions_hybrid.npz"
        np.savez_compressed(hybrid_preds_path, y_true=y_test, y_pred_probs=hybrid_pred_probs)
        self.logger.info(f"Test set hybrid predictions saved to {hybrid_preds_path.resolve()}")
        
        # Compute Hybrid Metrics
        hybrid_metrics = calculate_metrics(y_test, hybrid_pred_probs)
        hybrid_metrics["training_time"] = float(hybrid_train_time)
        hybrid_metrics["inference_time"] = float(hybrid_inference_total_time)
        
        # Save final metrics summary to CSV
        df_metrics = pd.DataFrame([cnn_metrics])
        metrics_csv_path = self.config.METRICS_DIR / "evaluation_metrics.csv"
        df_metrics.to_csv(metrics_csv_path, index=False)
        self.logger.info(f"Saved test evaluation metrics to {metrics_csv_path.name}")
        
        # Save model comparison metrics
        comparison_records = []
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "training_time", "inference_time"]:
            comparison_records.append({
                "Metric": key.upper().replace("_", " "),
                "CNN Only": cnn_metrics[key],
                "CNN + LightGBM": hybrid_metrics[key]
            })
        
        df_comparison = pd.DataFrame(comparison_records)
        comparison_csv_path = self.config.METRICS_DIR / "model_comparison_metrics.csv"
        df_comparison.to_csv(comparison_csv_path, index=False)
        self.logger.info(f"Saved model comparison metrics to {comparison_csv_path.name}")
        
        # 6. Decoupled Plotting Visualization
        plot_diagnostic_curves(history, self.config.FIGURES_DIR)
        plot_roc_and_confusion_matrix(y_true, y_pred_probs, self.config.FIGURES_DIR)
        
        # Save comparison plots
        from src.visualization.plots import plot_model_comparison
        plot_model_comparison(y_test, y_pred_probs, hybrid_pred_probs, cnn_metrics, hybrid_metrics, self.config.FIGURES_DIR)
        
        # 7. Export Experiment Summary JSON to summaries/
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
            "final_metrics": cnn_metrics,
            "hybrid_metrics": hybrid_metrics,
            "saved_model_paths": {
                "best_model": str((self.config.MODELS_DIR / "best_model.pt").resolve()),
                "final_model": str((self.config.MODELS_DIR / "final_model.pt").resolve()),
                "cnn_encoder": str(cnn_encoder_path.resolve()),
                "lightgbm_model": str(lgb_path.resolve())
            },
            "figure_paths": [
                str((self.config.FIGURES_DIR / "loss_accuracy_curves.png").resolve()),
                str((self.config.FIGURES_DIR / "roc_confusion_matrix.png").resolve()),
                str((self.config.FIGURES_DIR / "model_comparison_plots.png").resolve())
            ],
            "training_duration_seconds": float(duration)
        }
        
        summary_path = self.config.SUMMARIES_DIR / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"Experiment summary telemetry exported to {summary_path.resolve()}")
        
        # 8. Post-experiment automated checks validation
        self.validate_experiment_outputs()
        
        return summary

    def _extract_embeddings_and_labels(self, model, loader):
        model.eval()
        embeddings_list = []
        labels_list = []
        stellar_list = []
        with torch.no_grad():
            for x_global, x_local, x_stellar, y_true_batch in loader:
                x_global = x_global.to(self.config.DEVICE)
                x_local = x_local.to(self.config.DEVICE)
                x_stellar = x_stellar.to(self.config.DEVICE)
                
                # Penultimate embedding (size 128)
                emb = model.get_penultimate_embedding(x_global, x_local, x_stellar)
                
                embeddings_list.append(emb.cpu().numpy())
                labels_list.append(y_true_batch.numpy())
                stellar_list.append(x_stellar.cpu().numpy())
                
        # Stack all batches
        embeddings = np.concatenate(embeddings_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        stellar = np.concatenate(stellar_list, axis=0)
        
        # Concatenate: CNN embedding + stellar parameters -> 131 dimensions
        fused_features = np.concatenate((embeddings, stellar), axis=1)
        return fused_features, labels

    def validate_experiment_outputs(self) -> dict:
        """
        Validates that all expected artifact files were successfully written to disk.
        Returns a dictionary mapping path checks to binary outcomes.
        """
        checks = {
            "best_model_checkpoint": (self.config.MODELS_DIR / "best_model.pt").exists(),
            "final_model_checkpoint": (self.config.MODELS_DIR / "final_model.pt").exists(),
            "cnn_encoder_checkpoint": (self.config.MODELS_DIR / "cnn_encoder.pt").exists(),
            "lightgbm_model_txt": (self.config.MODELS_DIR / "lightgbm_model.txt").exists(),
            "metrics_history_csv": (self.config.METRICS_DIR / "history.csv").exists(),
            "evaluation_metrics_csv": (self.config.METRICS_DIR / "evaluation_metrics.csv").exists(),
            "model_comparison_metrics_csv": (self.config.METRICS_DIR / "model_comparison_metrics.csv").exists(),
            "loss_accuracy_curves": (self.config.FIGURES_DIR / "loss_accuracy_curves.png").exists(),
            "roc_confusion_matrix": (self.config.FIGURES_DIR / "roc_confusion_matrix.png").exists(),
            "model_comparison_plots": (self.config.FIGURES_DIR / "model_comparison_plots.png").exists(),
            "config_snapshot_json": (self.config.CONFIGS_DIR / "config.json").exists(),
            "experiment_summary_json": (self.config.SUMMARIES_DIR / "experiment_summary.json").exists(),
            "model_summary_txt": (self.config.SUMMARIES_DIR / "model_summary.txt").exists(),
            "predictions_npz": (self.config.PREDICTIONS_DIR / "test_predictions.npz").exists(),
            "predictions_hybrid_npz": (self.config.PREDICTIONS_DIR / "test_predictions_hybrid.npz").exists(),
            "feature_embeddings_npz": (self.config.PREDICTIONS_DIR / "feature_embeddings.npz").exists(),
            "training_log": (self.config.LOGS_DIR / "training.log").exists()
        }
        
        print("\n" + "=" * 60)
        print("            POST-EXPERIMENT ARTIFACT VALIDATION")
        print("=" * 60)
        all_passed = True
        for key, exists in checks.items():
            status = "PASSED" if exists else "FAILED"
            if not exists:
                all_passed = False
            print(f"{key:<30}: {status}")
        print("=" * 60)
        print(f"OVERALL VALIDATION STATUS     : {'SUCCESS' if all_passed else 'FAILURE'}")
        print("=" * 60 + "\n")
        
        return checks

