"""Synthetic spatiotemporal dataset generator."""
import torch
import numpy as np
from typing import Tuple


def generate_synthetic_data(
    num_samples: int,
    num_nodes: int,
    num_features: int,
    seq_len: int,
    pred_len: int,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic spatiotemporal data.
    
    Args:
        num_samples: Number of samples to generate
        num_nodes: Number of nodes in the graph
        num_features: Number of features per node
        seq_len: Input sequence length
        pred_len: Prediction sequence length
        seed: Random seed
        
    Returns:
        Tuple of (x, y) where:
            x: [num_samples, seq_len, num_nodes, num_features]
            y: [num_samples, pred_len, num_nodes, 1]
    """
    print(f"Generating synthetic data with num_samples: {num_samples}, num_nodes: {num_nodes}, num_features: {num_features}, seq_len: {seq_len}, pred_len: {pred_len}, seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    total_len = seq_len + pred_len
    
    # Generate temporal patterns with some spatial correlation
    x_list = []
    y_list = []
    
    for _ in range(num_samples):
        # Generate base signal with trend and seasonality
        time_steps = np.arange(total_len)
        
        # Create features with different patterns
        features = []
        for f in range(num_features):
            # Trend component
            trend = 0.1 * time_steps
            
            # Seasonal component
            seasonal = 5 * np.sin(2 * np.pi * time_steps / 12)
            
            # Random noise
            noise = np.random.randn(total_len, num_nodes) * 0.5
            
            # Spatial correlation (nodes closer together have similar patterns)
            spatial_corr = np.random.randn(num_nodes) * 2
            spatial_pattern = np.outer(np.ones(total_len), spatial_corr)
            
            feature = trend[:, None] + seasonal[:, None] + noise + spatial_pattern
            features.append(feature)
        
        # Stack features: [total_len, num_nodes, num_features]
        sample = np.stack(features, axis=-1)
        
        # Split into input and target
        x_sample = sample[:seq_len]  # [seq_len, num_nodes, num_features]
        y_sample = sample[seq_len:, :, 0:1]  # [pred_len, num_nodes, 1] (only first feature)
        
        x_list.append(x_sample)
        y_list.append(y_sample)
    
    x = torch.FloatTensor(np.stack(x_list))  # [num_samples, seq_len, num_nodes, num_features]
    y = torch.FloatTensor(np.stack(y_list))  # [num_samples, pred_len, num_nodes, 1]
    
    return x, y

