from typing import Dict, Type
import torch.nn as nn
from src.models.astronet import AstroNet

# Registry mapping model name to class constructors
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "astronet": AstroNet
}

def register_model(name: str, model_cls: Type[nn.Module]) -> None:
    """
    Registers a new model class in the registry.
    
    Args:
        name: Name identifier for the model.
        model_cls: Class constructor.
    """
    _MODEL_REGISTRY[name.lower()] = model_cls

def get_model(name: str, **kwargs) -> nn.Module:
    """
    Retrieve and instantiate a model from the registry by its identifier.
    
    Args:
        name: Model identifier string.
        kwargs: Hyperparameters or configuration options for the model constructor.
        
    Returns:
        nn.Module: Instantiated PyTorch model.
    """
    name_clean = name.lower()
    if name_clean not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name_clean](**kwargs)
