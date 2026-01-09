"""Wrapper for TGCN model to adapt it to the benchmark interface."""
import torch
import torch.nn as nn
from models.tgcn.original import TGCN


class TGCNWrapper(nn.Module):
    """Wrapper that adapts TGCN to benchmark interface.
    
    Handles:
    - Multi-feature input projection (F → 1)
    - Single-step to multi-step forecasting via autoregressive rollout
    - Output shape conversion [B, N, hidden_dim] → [B, Tout, N, 1]
    
    Note: Adjacency matrix is expected to be already normalized.
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # Feature projection: F → 1 (if F > 1)
        if num_features > 1:
            self.feature_proj = nn.Linear(num_features, 1)
        else:
            self.feature_proj = None
        
        # Placeholder for adj - will be set in forward
        # We can't initialize TGCN here because we need the laplacian from adj
        self.tgcn = None
        
        # Output head: hidden_dim → 1
        self.output_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, Tin, N, F]
            adj: Pre-normalized adjacency matrix [N, N] (used directly without further normalization)
            
        Returns:
            Predictions [B, Tout, N, 1]
        """
        if adj is None:
            raise ValueError("TGCN requires adjacency matrix")
        
        B, Tin, N, F = x.shape
        
        # Ensure adj is on same device as x (no normalization - adj is already normalized)
        adj = adj.to(x.device)
        
        # Initialize TGCN if not done yet (lazy initialization)
        if self.tgcn is None:
            # Pass pre-normalized adjacency directly (TGCN now accepts torch.Tensor and uses it without normalization)
            self.tgcn = TGCN(adj, hidden_dim=self.hidden_dim)
            # Ensure TGCN is on same device as x
            self.tgcn = self.tgcn.to(x.device)
        
        # Project features: [B, Tin, N, F] → [B, Tin, N, 1] if F > 1
        if self.feature_proj is not None:
            x = x.reshape(B * Tin * N, F)
            x = self.feature_proj(x)
            x = x.reshape(B, Tin, N, 1)
        
        # Squeeze feature dimension: [B, Tin, N, 1] → [B, Tin, N]
        x = x.squeeze(-1)
        
        # Process input sequence through TGCN to get final hidden state
        # TGCN expects [B, Tin, N] and outputs [B, N, hidden_dim]
        # But internally it processes sequence and returns final hidden state
        final_hidden = self.tgcn(x)  # [B, N, hidden_dim]
        
        # Get the TGCN cell for autoregressive rollout
        tgcn_cell = self.tgcn.tgcn_cell
        
        # Prepare for autoregressive rollout
        # TGCNCell expects hidden state as [B, N * hidden_dim]
        current_hidden = final_hidden.reshape(B, N * self.hidden_dim)
        current_input = x[:, -1, :]  # Last input step [B, N]
        
        predictions = []
        
        # Autoregressive rollout for multi-step forecasting
        for t in range(self.pred_len):
            # Get prediction from current hidden state
            hidden_reshaped = current_hidden.reshape(B, N, self.hidden_dim)
            step_pred = self.output_head(hidden_reshaped)  # [B, N, 1]
            predictions.append(step_pred)
            
            # Use prediction as input for next step (autoregressive)
            next_input = step_pred.squeeze(-1)  # [B, N]
            
            # Run TGCNCell one step to update hidden state
            _, current_hidden = tgcn_cell(next_input, current_hidden)
        
        # Stack predictions: [B, Tout, N, 1]
        predictions = torch.stack(predictions, dim=1)  # [B, Tout, N, 1]
        
        return predictions

