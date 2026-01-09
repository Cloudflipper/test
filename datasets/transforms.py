"""Data transforms and preprocessing."""
import torch
from typing import Tuple
from utils.scaler import StandardScaler


class DataTransform:
    """Data transformation pipeline."""
    
    def __init__(self, scaler: StandardScaler = None):
        self.scaler = scaler or StandardScaler()
        self.fitted = False
    
    def fit(self, data: torch.Tensor):
        """Fit scaler on training data."""
        self.scaler.fit(data)
        self.fitted = True
    
    def transform(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform input and target data.
        
        Args:
            x: Input tensor [B, Tin, N, F]
            y: Target tensor [B, Tout, N, 1]
            
        Returns:
            Scaled (x, y)
        """
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")
        
        x_scaled = self.scaler.transform(x)
        # For y, use only the first feature's mean and std (y has shape [B, Tout, N, 1])
        # Extract first feature's statistics
        y_mean = self.scaler.mean[..., 0:1]  # [1, 1]
        y_std = self.scaler.std[..., 0:1]    # [1, 1]
        y_scaled = (y - y_mean) / y_std
        
        return x_scaled, y_scaled
    
    def inverse_transform_y(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transform predictions back to original scale.
        
        Args:
            y: Scaled predictions [B, Tout, N, 1]
            
        Returns:
            Inverse scaled predictions [B, Tout, N, 1]
        """
        # Use only the first feature's mean and std for inverse transform
        y_mean = self.scaler.mean[..., 0:1]  # [1, 1]
        y_std = self.scaler.std[..., 0:1]    # [1, 1]
        return y * y_std + y_mean

