"""Evaluation metrics for spatiotemporal forecasting."""
import torch
import numpy as np


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(y_true - y_pred)).item()


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def mape(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    mask = torch.abs(y_true) > epsilon
    if mask.sum() == 0:
        return 0.0
    return (torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100).mean().item()


def evaluate(y_true: torch.Tensor, y_pred: torch.Tensor, metrics: list = None) -> dict:
    """Evaluate predictions against ground truth.
    
    Args:
        y_true: Ground truth tensor [B, Tout, N, 1]
        y_pred: Predicted tensor [B, Tout, N, 1]
        metrics: List of metric names to compute
        
    Returns:
        Dictionary of metric values
    """
    if metrics is None:
        metrics = ["MAE", "RMSE", "MAPE"]
    
    results = {}
    
    for metric_name in metrics:
        if metric_name == "MAE":
            results["MAE"] = mae(y_true, y_pred)
        elif metric_name == "RMSE":
            results["RMSE"] = rmse(y_true, y_pred)
        elif metric_name == "MAPE":
            results["MAPE"] = mape(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    return results

