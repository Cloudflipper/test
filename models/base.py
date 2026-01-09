"""Base interface for spatiotemporal forecasting models."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseSpatioTemporalModel(nn.Module, ABC):
    """Base class for spatiotemporal forecasting models.
    
    All models should implement this interface to ensure compatibility
    with the benchmark infrastructure.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [B, Tin, N, F]
               - B: batch size
               - Tin: input sequence length
               - N: number of nodes
               - F: number of features
            adj: Optional adjacency matrix of shape [N, N]
                 Some models may require this, others may ignore it.
                 
        Returns:
            Predictions tensor of shape [B, Tout, N, 1]
            - B: batch size
            - Tout: prediction sequence length
            - N: number of nodes
            - 1: single feature (target variable)
        """
        pass

