# models/tgcn_direct/original.py
import torch
import torch.nn as nn

from models.tgcn.original import TGCNCell


class TGCNEncoder(nn.Module):
    """
    TGCN encoder (no rollout, no attention).
    Matches TF baseline idea: run TGCNCell over Tin steps, take last hidden.

    Input:  x [B, Tin, N]
    Output: last_hidden [B, N, H]
    """
    def __init__(self, adj: torch.Tensor, hidden_dim: int = 64):
        super().__init__()
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        self.register_buffer("adj", adj)

        self.num_nodes = adj.shape[0]
        self.hidden_dim = hidden_dim

        # TGCNCell expects input_dim == num_nodes (scalar per node)
        self.cell = TGCNCell(self.adj, input_dim=self.num_nodes, hidden_dim=self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Tin, N]
        B, Tin, N = x.shape
        assert N == self.num_nodes, f"num_nodes mismatch: x has {N}, adj has {self.num_nodes}"

        h = torch.zeros(B, N * self.hidden_dim, device=x.device, dtype=x.dtype)
        out = None
        for t in range(Tin):
            out, h = self.cell(x[:, t, :], h)  # out: [B, N*H]
        out = out.reshape(B, N, self.hidden_dim)  # [B, N, H]
        return out
