# models/ktgcn/original.py
import os
import numpy as np
import torch
import torch.nn as nn

from models.tgcn.original import TGCNCell


class UnitStatic(nn.Module):
    """
    PyTorch equivalent of TF Unit_static in ktgcn.py.

    Given:
      x: [B, N]
      E: [N, D]  (KG embedding)
    It computes:
      u = relu((x @ E) @ W + b)   -> [B, N]
    """
    def __init__(self, num_nodes: int, kg_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.kg_dim = kg_dim
        self.weight_unit = nn.Parameter(torch.empty(kg_dim, num_nodes))
        self.bias_unit = nn.Parameter(torch.zeros(num_nodes))
        nn.init.xavier_uniform_(self.weight_unit)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        # x: [B, N], E: [N, D]
        # unit_matrix: [B, D]
        unit_matrix = x @ E
        # back to nodes: [B, N]
        out = unit_matrix @ self.weight_unit + self.bias_unit
        return self.act(out)


def load_kg_embedding_csv(path: str, device: torch.device, dtype: torch.dtype):
    """
    Load kg embedding CSV and apply the same normalization style as TF code:
      kgembedding_nor = abs(kgembedding / max(kgembedding))
      dropna(axis=1)
    Returns: E [N, D]
    """
    import pandas as pd

    df = pd.read_csv(path, header=None)
    max_val = float(np.nanmax(df.values))
    if max_val == 0 or np.isnan(max_val):
        raise ValueError(f"Invalid KG embedding max value from {path}: {max_val}")

    df = df / max_val
    df = df.abs()
    df = df.dropna(axis=1)

    E = torch.tensor(df.values, device=device, dtype=dtype)
    return E


class KTGCNEncoder(nn.Module):
    """
    Encode Tin steps using:
      x_t --(UnitStatic w/ KG emb)--> x'_t --(TGCNCell)--> hidden

    Input:  x [B, Tin, N]
    Output: last_hidden [B, N, H]
    """
    def __init__(
        self,
        adj: torch.Tensor,
        hidden_dim: int,
        kg_emb_path: str,
        kg_dim: int,
    ):
        super().__init__()
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        self.register_buffer("adj", adj)

        self.num_nodes = adj.shape[0]
        self.hidden_dim = hidden_dim

        self.kg_emb_path = kg_emb_path
        self.kg_dim = kg_dim

        # created lazily on first forward when we know device/dtype
        self.register_buffer("_E", torch.empty(0), persistent=False)
        self.unit_static = UnitStatic(num_nodes=self.num_nodes, kg_dim=self.kg_dim)

        # base TGCN cell (scalar per node)
        self.cell = TGCNCell(self.adj, input_dim=self.num_nodes, hidden_dim=self.hidden_dim)

    def _maybe_load_E(self, device: torch.device, dtype: torch.dtype):
        if self._E.numel() == 0:
            if not self.kg_emb_path or not os.path.exists(self.kg_emb_path):
                raise FileNotFoundError(
                    f"KG embedding CSV not found: {self.kg_emb_path}. "
                    f"Provide model.kg_emb_path in config."
                )
            E = load_kg_embedding_csv(self.kg_emb_path, device=device, dtype=dtype)

            # sanity: first dim must be N
            if E.shape[0] != self.num_nodes:
                raise ValueError(
                    f"KG embedding N mismatch: E.shape[0]={E.shape[0]} vs num_nodes={self.num_nodes}"
                )
            if E.shape[1] != self.kg_dim:
                # allow auto-infer if user didn't set kg_dim correctly
                # but keep strict by default
                raise ValueError(
                    f"KG dim mismatch: E.shape[1]={E.shape[1]} vs kg_dim={self.kg_dim}. "
                    f"Set model.kg_dim accordingly."
                )

            self._E = E  # buffer on correct device/dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Tin, N]
        B, Tin, N = x.shape
        assert N == self.num_nodes, f"num_nodes mismatch: x has {N}, adj has {self.num_nodes}"

        self._maybe_load_E(device=x.device, dtype=x.dtype)

        h = torch.zeros(B, N * self.hidden_dim, device=x.device, dtype=x.dtype)
        out = None
        for t in range(Tin):
            x_t = x[:, t, :]                 # [B, N]
            x_t = self.unit_static(x_t, self._E)  # [B, N]  (KG-informed input)
            out, h = self.cell(x_t, h)       # out: [B, N*H]
        out = out.reshape(B, N, self.hidden_dim)
        return out
