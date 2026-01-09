# models/tgcn_att/wrapper.py
import torch
import torch.nn as nn

from models.tgcn_att.original import TGCNAtt


class TGCNAttWrapper(nn.Module):
    """
    Benchmark interface wrapper.

    forward(x, adj) -> [B, Tout, N, 1]
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
        att_hidden_dim: int = 32,
        att_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Feature projection: F -> 1 (if F > 1)
        if num_features > 1:
            self.feature_proj = nn.Linear(num_features, 1)
        else:
            self.feature_proj = None

        # Lazy init (needs adj)
        self.tgcn_att = None

        # hidden -> 1
        self.output_head = nn.Linear(hidden_dim, 1)

        # store att params for init
        self._att_hidden_dim = att_hidden_dim
        self._att_dropout = att_dropout

    def _maybe_init(self, adj: torch.Tensor, device: torch.device):
        if self.tgcn_att is None:
            self.tgcn_att = TGCNAtt(
                adj=adj,
                hidden_dim=self.hidden_dim,
                att_hidden_dim=self._att_hidden_dim,
                att_dropout=self._att_dropout,
            ).to(device)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        if adj is None:
            raise ValueError("TGCNAtt requires adjacency matrix")

        B, Tin, N, F = x.shape
        if N != self.num_nodes:
            # not fatal but indicates config/dataloader mismatch
            raise ValueError(f"num_nodes mismatch: wrapper num_nodes={self.num_nodes}, x has N={N}")

        adj = adj.to(x.device)
        self._maybe_init(adj, x.device)

        # F -> 1 if needed
        if self.feature_proj is not None:
            x_ = x.reshape(B * Tin * N, F)
            x_ = self.feature_proj(x_)
            x = x_.reshape(B, Tin, N, 1)

        # squeeze feature dim -> [B, Tin, N]
        x_in = x.squeeze(-1)

        # Encode with attention
        # context: [B, N, H], alpha: [B, Tin, N]
        context, last_hidden_state, alpha = self.tgcn_att(x_in)

        # Use attention context as initial hidden for rollout
        current_hidden = context.reshape(B, N * self.hidden_dim)
        tgcn_cell = self.tgcn_att.tgcn_cell

        # Start autoregressive rollout from last observed input
        current_input = x_in[:, -1, :]  # [B, N]

        preds = []
        for _ in range(self.pred_len):
            hidden_reshaped = current_hidden.reshape(B, N, self.hidden_dim)
            step_pred = self.output_head(hidden_reshaped)  # [B, N, 1]
            preds.append(step_pred)

            next_input = step_pred.squeeze(-1)  # [B, N]
            _, current_hidden = tgcn_cell(next_input, current_hidden)

        return torch.stack(preds, dim=1)  # [B, Tout, N, 1]
