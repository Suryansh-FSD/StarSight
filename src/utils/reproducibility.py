import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for PyTorch, NumPy, and random packages to ensure reproducibility.
    
    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"Reproducible random seed set to: {seed}")
