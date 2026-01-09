# models/ktgcn/wrapper.py
import torch
import torch.nn as nn

from models.ktgcn.original import KTGCNEncoder


class KTGCNWrapper(nn.Module):
    """
    Benchmark interface:
      forward(x, adj) -> [B, Tout, N, 1]
    x:   [B, Tin, N, F]
    adj: [N, N]
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 64,
        kg_emb_path: str = "",
        kg_dim: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        self.kg_emb_path = kg_emb_path
        self.kg_dim = kg_dim

        # multi-channel -> 1
        self.feature_proj = nn.Linear(num_features, 1) if num_features > 1 else None

        # lazy init (needs adj)
        self.encoder = None

        # direct head: hidden -> pred_len
        self.output_head = nn.Linear(hidden_dim, pred_len)

    def _maybe_init(self, adj: torch.Tensor, device: torch.device):
        if self.encoder is None:
            self.encoder = KTGCNEncoder(
                adj=adj,
                hidden_dim=self.hidden_dim,
                kg_emb_path=self.kg_emb_path,
                kg_dim=self.kg_dim,
            ).to(device)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        if adj is None:
            raise ValueError("KTGCNWrapper requires adjacency matrix")

        B, Tin, N, F = x.shape
        if N != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: wrapper num_nodes={self.num_nodes}, x has N={N}")
        if Tin != self.seq_len:
            raise ValueError(f"seq_len mismatch: wrapper seq_len={self.seq_len}, x has Tin={Tin}")

        adj = adj.to(x.device)
        self._maybe_init(adj, x.device)

        # F -> 1 if needed
        if self.feature_proj is not None:
            x_flat = x.reshape(B * Tin * N, F)
            x_flat = self.feature_proj(x_flat)
            x = x_flat.reshape(B, Tin, N, 1)

        x_in = x.squeeze(-1)  # [B, Tin, N]

        last_hidden = self.encoder(x_in)     # [B, N, H]
        y = self.output_head(last_hidden)    # [B, N, pred_len]
        y = y.permute(0, 2, 1).unsqueeze(-1) # [B, pred_len, N, 1]
        return y
