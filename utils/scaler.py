"""Data scaling utilities."""
import torch
import numpy as np
from typing import Optional


class StandardScaler:
    """Standard scaler that fits on training data only."""
    
    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
    
    def fit(self, data: torch.Tensor):
        """Fit scaler on training data.
        
        Args:
            data: Tensor of shape [B, T, N, F] or [T, N, F]
        """
        if data.dim() == 4:
            # [B, T, N, F] -> flatten to [B*T*N, F]
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            # [T, N, F] -> flatten to [T*N, F]
            data_flat = data.reshape(-1, data.shape[-1])
        
        self.mean = torch.mean(data_flat, dim=0, keepdim=True)
        self.std = torch.std(data_flat, dim=0, keepdim=True)
        # Avoid division by zero
        self.std = torch.clamp(self.std, min=1e-8)
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform data back to original scale."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return data * self.std + self.mean

