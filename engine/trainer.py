"""Training engine."""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
from utils.logger import setup_logger
from utils.metrics import evaluate


class Trainer:
    """Trainer for spatiotemporal forecasting models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        adj: torch.Tensor,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str,
        log_dir: str,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0001,
        gradient_clip: float = 5.0,
        logger=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.adj = adj.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.gradient_clip = gradient_clip
        self.logger = logger or setup_logger("trainer", log_dir)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for x, y in self.train_loader:
            x = x.to(self.device)  # [B, Tin, N, F]
            y = y.to(self.device)  # [B, Tout, N, 1]
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(x, self.adj)  # [B, Tout, N, 1]
            
            # Loss
            loss = self.criterion(pred, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> tuple:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(x, self.adj)
                loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = evaluate(all_targets, all_preds)
        
        return avg_loss, metrics
    
    def train(self, epochs: int):
        """Train model for specified number of epochs."""
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Start total training timer
        total_start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch + 1
            
            # Start epoch timer
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Logging with timing
            self.logger.info(
                f"Epoch {self.epoch}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Val MAE: {val_metrics['MAE']:.6f}, "
                f"Val RMSE: {val_metrics['RMSE']:.6f}, "
                f"Val MAPE: {val_metrics['MAPE']:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping and checkpointing
            improved = val_loss < (self.best_val_loss - self.early_stopping_min_delta)
            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {self.epoch}")
                break
        
        # Calculate total training time
        total_time = time.time() - total_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        self.logger.info(f"Training completed. Best val loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Total training time: {hours}h {minutes}m {seconds}s ({total_time:.2f}s)")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")

