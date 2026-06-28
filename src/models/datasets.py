import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

class StarSightDataset(Dataset):
    """
    Custom PyTorch Dataset for loading AstroNet exoplanet detection features.
    """
    def __init__(self, X_global: np.ndarray, X_local: np.ndarray, X_stellar: np.ndarray, y: np.ndarray):
        self.X_global = torch.tensor(X_global, dtype=torch.float32)
        self.X_local = torch.tensor(X_local, dtype=torch.float32)
        self.X_stellar = torch.tensor(X_stellar, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Align dimensions to (Batch, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X_global[idx], self.X_local[idx], self.X_stellar[idx], self.y[idx]


def create_data_loaders(data_path: Path, config) -> tuple:
    """
    Load data arrays from compressed file, partition indices, export data splits,
    and return Train, Validation, and Test PyTorch DataLoader instances.
    
    Args:
        data_path: Path to the final_dataset.npz file.
        config: Project configurations object.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset)
    """
    # Load dataset
    data = np.load(data_path)
    X_global = data['X_global']
    X_local = data['X_local']
    X_stellar = data['X_stellar']
    y = data['y']
    
    # If in DEV_MODE, truncate dataset to 4 samples for rapid verification
    if config.DEV_MODE:
        X_global = X_global[:4]
        X_local = X_local[:4]
        X_stellar = X_stellar[:4]
        y = y[:4]
    
    num_samples = len(y)
    indices = np.arange(num_samples)
    
    # Load exoplanet target details mapping
    csv_path = config.RESULTS_DIR / "transit_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        base_names = list(df["Target"])
        target_names = []
        for name in base_names:
            target_names.append(f"{name}_Pos")
            target_names.append(f"{name}_Neg")
        if config.DEV_MODE:
            target_names = target_names[:4]
    else:
        target_names = [f"Target_{i}" for i in range(num_samples)]

        
    # Split splits with fixed seed
    # Handling small datasets (e.g. 4 samples in dev mode)
    if num_samples < 5:
        # Avoid splitting failures on extremely small tests
        train_idx = indices[:max(1, int(0.5 * num_samples))]
        val_idx = indices[len(train_idx):max(len(train_idx)+1, int(0.75 * num_samples))]
        test_idx = indices[len(train_idx)+len(val_idx):]
    else:
        # Standard random split
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.4, random_state=config.RANDOM_SEED, stratify=y
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=config.RANDOM_SEED, stratify=y[temp_idx]
        )
        
    # Translate split indices into target names
    split_info = {
        "train": [target_names[i] for i in train_idx],
        "validation": [target_names[i] for i in val_idx],
        "test": [target_names[i] for i in test_idx]
    }
    
    # Export split information
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_DIR / "data_split.json", "w") as f:
        json.dump(split_info, f, indent=4)
        
    # Initialize PyTorch datasets
    train_dataset = StarSightDataset(X_global[train_idx], X_local[train_idx], X_stellar[train_idx], y[train_idx])
    val_dataset = StarSightDataset(X_global[val_idx], X_local[val_idx], X_stellar[val_idx], y[val_idx])
    test_dataset = StarSightDataset(X_global[test_idx], X_local[test_idx], X_stellar[test_idx], y[test_idx])
    
    # Dynamic adaptive batch sizing: min(32, len(train_dataset))
    adaptive_batch_size = min(32, len(train_dataset))
    
    # Initialize DataLoader streams
    train_loader = DataLoader(train_dataset, batch_size=adaptive_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=adaptive_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=adaptive_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset
