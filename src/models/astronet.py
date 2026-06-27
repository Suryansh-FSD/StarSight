import torch
import torch.nn as nn
from src.models.base_model import StarSightBaseModel

class AstroNet(StarSightBaseModel):
    """
    A dual-branch Convolutional Neural Network inspired by AstroNet for exoplanet detection.
    Processes a high-dimensional Global View and a zoomed-in Local View, concatenating 
    them with stellar parameters to output exoplanet candidate logits.
    """
    def __init__(self, global_in_size: int = 2001, local_in_size: int = 201, stellar_in_size: int = 3):
        super(AstroNet, self).__init__()
        
        # 1. Global View Branch
        # Input shape: (Batch, 1, 2001)
        self.global_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # Fixes dimension to (Batch, 128, 8) -> 1024 flat
        )
        
        # 2. Local View Branch
        # Input shape: (Batch, 1, 201)
        self.local_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # Fixes dimension to (Batch, 64, 8) -> 512 flat
        )
        
        # 3. Dense Classifier Head
        # Concatenated features size: 1024 (global) + 512 (local) + 3 (stellar) = 1539
        concat_size = 128 * 8 + 64 * 8 + stellar_in_size
        
        self.classifier = nn.Sequential(
            nn.Linear(concat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Outputs logits for BCEWithLogitsLoss
        )

    def forward(self, x_global: torch.Tensor, x_local: torch.Tensor, x_stellar: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing CNN branches, fusion, and fully connected classification.
        """
        # Ensure dimensions match (Batch, Channels=1, Length)
        if x_global.dim() == 2:
            x_global = x_global.unsqueeze(1)
        if x_local.dim() == 2:
            x_local = x_local.unsqueeze(1)
            
        # Extract features
        g_feat = self.global_branch(x_global)
        g_feat = g_feat.view(g_feat.size(0), -1)  # Flatten
        
        l_feat = self.local_branch(x_local)
        l_feat = l_feat.view(l_feat.size(0), -1)  # Flatten
        
        # Concatenate features
        fused = torch.cat((g_feat, l_feat, x_stellar), dim=1)
        
        # Output logits
        return self.classifier(fused)
