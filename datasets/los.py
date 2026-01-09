"""LOS (Los Angeles) traffic dataset loader."""
import numpy as np
import pandas as pd
import torch
from typing import Tuple
import os


def load_los_data(
    data_dir: str = "./data",
    seq_len: int = 12,
    pred_len: int = 12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load LOS traffic dataset.
    
    Args:
        data_dir: Directory containing los_adj.csv and los_speed.csv
        seq_len: Input sequence length
        pred_len: Prediction sequence length
        
    Returns:
        Tuple of (x, y) where:
            x: [num_samples, seq_len, num_nodes, 1] - input sequences
            y: [num_samples, pred_len, num_nodes, 1] - target sequences
    """
    # Load speed data: [T, N]
    speed_path = os.path.join(data_dir, "los_speed.csv")
    speed_data = pd.read_csv(speed_path, header=None).values  # [T, N]
    
    T, N = speed_data.shape
    
    # Convert to torch tensor
    speed_tensor = torch.FloatTensor(speed_data)  # [T, N]
    
    # Create sliding windows
    num_samples = T - seq_len - pred_len + 1
    
    x_list = []
    y_list = []
    
    for i in range(num_samples):
        # Input: [seq_len, N]
        x_sample = speed_tensor[i:i+seq_len, :]  # [seq_len, N]
        # Target: [pred_len, N]
        y_sample = speed_tensor[i+seq_len:i+seq_len+pred_len, :]  # [pred_len, N]
        
        # Add feature dimension: [seq_len, N] -> [seq_len, N, 1]
        x_sample = x_sample.unsqueeze(-1)  # [seq_len, N, 1]
        y_sample = y_sample.unsqueeze(-1)  # [pred_len, N, 1]
        
        x_list.append(x_sample)
        y_list.append(y_sample)
    
    # Stack: [num_samples, seq_len, N, 1] and [num_samples, pred_len, N, 1]
    x = torch.stack(x_list, dim=0)  # [num_samples, seq_len, N, 1]
    y = torch.stack(y_list, dim=0)  # [num_samples, pred_len, N, 1]
    
    return x, y

