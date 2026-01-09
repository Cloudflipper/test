# models/tgcn_direct/wrapper.py
import torch
import torch.nn as nn

from models.tgcn_direct.original import TGCNEncoder


class TGCNDirectWrapper(nn.Module):
    """
    Benchmark interface:
      forward(x, adj) -> [B, Tout, N, 1]
    where:
      x:   [B, Tin, N, F]
      adj: [N, N]  (already normalized)
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Feature fusion if F>1
        self.feature_proj = nn.Linear(num_features, 1) if num_features > 1 else None

        # Lazy init because encoder needs adj
        self.encoder = None

        # Direct multi-step head: hidden -> pred_len
        # TF baseline: last_hidden @ W + b gives pre_len
        self.output_head = nn.Linear(hidden_dim, pred_len)

    def _maybe_init(self, adj: torch.Tensor, device: torch.device):
        if self.encoder is None:
            self.encoder = TGCNEncoder(adj=adj, hidden_dim=self.hidden_dim).to(device)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        if adj is None:
            raise ValueError("TGCNDirectWrapper requires adjacency matrix")

        B, Tin, N, F = x.shape
        if N != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: wrapper num_nodes={self.num_nodes}, x has N={N}")
        if Tin != self.seq_len:
            # not fatal, but helps catch config mistakes
            raise ValueError(f"seq_len mismatch: wrapper seq_len={self.seq_len}, x has Tin={Tin}")

        adj = adj.to(x.device)
        self._maybe_init(adj, x.device)

        # F -> 1 if needed
        if self.feature_proj is not None:
            x_flat = x.reshape(B * Tin * N, F)
            x_flat = self.feature_proj(x_flat)           # [B*Tin*N, 1]
            x = x_flat.reshape(B, Tin, N, 1)

        x_in = x.squeeze(-1)  # [B, Tin, N]

        last_hidden = self.encoder(x_in)  # [B, N, H]

        # Direct multi-step output: [B, N, Tout]
        y = self.output_head(last_hidden)  # [B, N, pred_len]

        # Convert to benchmark output: [B, Tout, N, 1]
        y = y.permute(0, 2, 1).unsqueeze(-1)  # [B, pred_len, N, 1]
        return y
