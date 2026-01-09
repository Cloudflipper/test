"""Data loading utilities."""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from datasets.synthetic import generate_synthetic_data
from datasets.los import load_los_data
from datasets.transforms import DataTransform


class SpatioTemporalDataset(Dataset):
    """Spatiotemporal forecasting dataset."""
    
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """Initialize dataset.
        
        Args:
            x: Input tensor [num_samples, seq_len, num_nodes, num_features]
            y: Target tensor [num_samples, pred_len, num_nodes, 1]
        """
        self.x = x
        self.y = y
    
    def __len__(self) -> int:
        return self.x.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def create_dataloaders(
    dataset_name: str = "synthetic",
    num_samples: Optional[int] = None,
    num_nodes: Optional[int] = None,
    num_features: Optional[int] = None,
    seq_len: int = 12,
    pred_len: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader, DataLoader, DataTransform]:
    """Create train/val/test dataloaders with proper scaling.
    
    Args:
        dataset_name: Name of dataset ("synthetic" or "los")
        num_samples: Number of samples (for synthetic only)
        num_nodes: Number of nodes (for synthetic only)
        num_features: Number of features (for synthetic only)
        seq_len: Input sequence length
        pred_len: Prediction sequence length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        batch_size: Batch size
        seed: Random seed
        data_dir: Data directory (for real datasets)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, transform)
    """
    # Load data based on dataset name
    if dataset_name == "synthetic":
        if num_samples is None or num_nodes is None or num_features is None:
            raise ValueError("num_samples, num_nodes, and num_features required for synthetic dataset")
        x, y = generate_synthetic_data(
            num_samples=num_samples,
            num_nodes=num_nodes,
            num_features=num_features,
            seq_len=seq_len,
            pred_len=pred_len,
            seed=seed
        )
    elif dataset_name == "los":
        x, y = load_los_data(data_dir=data_dir, seq_len=seq_len, pred_len=pred_len)
        # LOS dataset has F=1 (single feature)
        num_features = 1
        # Get num_samples from loaded data
        num_samples = x.shape[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split data
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val = x[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    # Fit scaler on training data only
    transform = DataTransform()
    transform.fit(x_train)
    
    # Transform all splits
    x_train_scaled, y_train_scaled = transform.transform(x_train, y_train)
    x_val_scaled, y_val_scaled = transform.transform(x_val, y_val)
    x_test_scaled, y_test_scaled = transform.transform(x_test, y_test)
    
    # Create datasets
    train_dataset = SpatioTemporalDataset(x_train_scaled, y_train_scaled)
    val_dataset = SpatioTemporalDataset(x_val_scaled, y_val_scaled)
    test_dataset = SpatioTemporalDataset(x_test_scaled, y_test_scaled)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, transform

