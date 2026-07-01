import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import torch
import src.config as config
from src.models.astronet import AstroNet
from src.models.lightgbm_classifier import LightGBMClassifier

def test_config():
    assert config.GLOBAL_BINS == 2001
    assert config.LOCAL_BINS == 201

def test_astronet_instantiation():
    model = AstroNet(
        global_in_size=config.GLOBAL_BINS,
        local_in_size=config.LOCAL_BINS,
        stellar_in_size=3
    )
    assert model is not None
    
    # Test forward pass with dummy tensors
    x_g = torch.randn(2, 1, config.GLOBAL_BINS)
    x_l = torch.randn(2, 1, config.LOCAL_BINS)
    x_s = torch.randn(2, 3)
    out = model(x_g, x_l, x_s)
    assert out.shape == (2, 1)

def test_lightgbm_wrapper():
    clf = LightGBMClassifier(n_estimators=5)
    assert clf is not None
