import time
import logging
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger("StarSight.Trainer")

def train_model(model: nn.Module, train_loader, val_loader, optimizer, criterion, config) -> list:
    """
    Train the AstroNet model with gradient clipping, early stopping,
    and checkpointing. Outputs model checkpoints and history logs.
    
    Returns:
        list: History records containing training and validation loss/accuracy.
    """
    device = config.DEVICE
    model.to(device)
    
    # Establish logging configuration
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / "training.log"
    
    # Reset logger handlers to prevent duplication
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] StarSight.Trainer - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger.info("Initializing AstroNet training run...")
    logger.info(f"Using device: {device}")
    logger.info(f"Hyperparameters: LR={config.LEARNING_RATE}, Epochs={config.MAX_EPOCHS}, Patience={config.EARLY_STOPPING_PATIENCE}")

    history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, config.MAX_EPOCHS + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_global, x_local, x_stellar, y_true in train_loader:
            x_global = x_global.to(device)
            x_local = x_local.to(device)
            x_stellar = x_stellar.to(device)
            y_true = y_true.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x_global, x_local, x_stellar)
            loss = criterion(logits, y_true)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * y_true.size(0)
            
            # Calculate predictions accuracy
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_correct += (preds == y_true).sum().item()
            train_total += y_true.size(0)
            
        epoch_train_loss = train_loss / train_total if train_total > 0 else 0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0
        
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_global, x_local, x_stellar, y_true in val_loader:
                x_global = x_global.to(device)
                x_local = x_local.to(device)
                x_stellar = x_stellar.to(device)
                y_true = y_true.to(device)
                
                logits = model(x_global, x_local, x_stellar)
                loss = criterion(logits, y_true)
                
                val_loss += loss.item() * y_true.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == y_true).sum().item()
                val_total += y_true.size(0)
                
        epoch_val_loss = val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0
        
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc
        }
        history.append(epoch_record)
        
        log_msg = (f"Epoch {epoch:02d}/{config.MAX_EPOCHS:02d} | "
                   f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
                   f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")
        logger.info(log_msg)
        print(log_msg)
        
        # Checkpointing (Best Model) using the base model helper save_checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_path = config.MODELS_DIR / "best_model.pt"
            model.save_checkpoint(best_model_path)
            logger.info(f"Validation loss improved. Saved best model to {best_model_path.name}")
        else:
            epochs_no_improve += 1
            
        # Early stopping check
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs due to validation stall.")
            print(f"Early stopping triggered after {epoch} epochs.")
            break
            
    # Save final model state using helper save_checkpoint
    final_model_path = config.MODELS_DIR / "final_model.pt"
    model.save_checkpoint(final_model_path)
    logger.info(f"Saved final model states to {final_model_path.name}")
    
    # Save history csv
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df_history = pd.DataFrame(history)
    history_csv_path = config.METRICS_DIR / "history.csv"
    df_history.to_csv(history_csv_path, index=False)
    logger.info(f"Saved epoch logs spreadsheet to {history_csv_path.name}")
    
    return history
