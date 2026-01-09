"""Evaluation engine."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from utils.metrics import evaluate
from utils.logger import setup_logger


class Evaluator:
    """Evaluator for spatiotemporal forecasting models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        adj: torch.Tensor,
        device: torch.device,
        transform,
        logger=None
    ):
        self.model = model
        self.test_loader = test_loader
        self.adj = adj.to(device)
        self.device = device
        self.transform = transform
        self.logger = logger or setup_logger("evaluator")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Returns:
            Dictionary of metrics computed on inverse-scaled predictions
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                pred = self.model(x, self.adj)  # [B, Tout, N, 1]
                
                # Inverse transform to original scale
                pred_original = self.transform.inverse_transform_y(pred.cpu())
                y_original = self.transform.inverse_transform_y(y.cpu())
                self.logger.info(f"pred(norm) range: {pred.min().item():.3f} ~ {pred.max().item():.3f}")
                self.logger.info(f"pred(real) range: {pred_original.min().item():.3f} ~ {pred_original.max().item():.3f}")
                self.logger.info(f"y(real) range: {y_original.min().item():.3f} ~ {y_original.max().item():.3f}")

                
                all_preds.append(pred_original)
                all_targets.append(y_original)
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics on original scale
        metrics = evaluate(all_targets, all_preds)
        
        self.logger.info("Test Results:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.6f}")
        
        return metrics

