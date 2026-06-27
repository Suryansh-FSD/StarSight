import torch
import torch.nn as nn
from typing import Tuple, list

def train_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Executes a single training epoch across the loader dataset.
    
    Returns:
        Tuple[float, float]: (average_loss, average_accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x_global, x_local, x_stellar, y_true in loader:
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
        
        total_loss += loss.item() * y_true.size(0)
        
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == y_true).sum().item()
        total += y_true.size(0)
        
    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    
    return avg_loss, avg_acc


def evaluate_epoch(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Executes a validation evaluation pass across the loader dataset.
    
    Returns:
        Tuple[float, float]: (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_global, x_local, x_stellar, y_true in loader:
            x_global = x_global.to(device)
            x_local = x_local.to(device)
            x_stellar = x_stellar.to(device)
            y_true = y_true.to(device)
            
            logits = model(x_global, x_local, x_stellar)
            loss = criterion(logits, y_true)
            
            total_loss += loss.item() * y_true.size(0)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y_true).sum().item()
            total += y_true.size(0)
            
    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    
    return avg_loss, avg_acc


def train_model(model: nn.Module, train_loader, val_loader, optimizer, criterion, config) -> list:
    """
    Legacy wrapper for backward compatibility with existing notebook code interfaces.
    Executes training but performs zero file writes.
    """
    history = []
    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, config.DEVICE)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
    return history
