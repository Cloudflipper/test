# models/tgcn_att/original.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tgcn.original import TGCNCell


class TemporalSelfAttention(nn.Module):
    """
    Time-wise attention over hidden sequence:
      h: [B, T, N, H]
      alpha: [B, T, N]  (softmax over T)
      context: [B, N, H]
    """
    def __init__(self, hidden_dim: int, att_hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, att_hidden_dim)
        self.fc2 = nn.Linear(att_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor):
        # h: [B, T, N, H]
        B, T, N, H = h.shape
        x = self.dropout(h)

        # score: [B, T, N, 1]
        score = self.fc2(torch.tanh(self.fc1(x)))

        # alpha over time dimension (T)
        alpha = F.softmax(score, dim=1)  # [B, T, N, 1]

        # context: [B, N, H]
        context = (alpha * h).sum(dim=1)

        return context, alpha.squeeze(-1)  # alpha: [B, T, N]


class TGCNAtt(nn.Module):
    """
    TGCN encoder with time-wise attention.

    Input:  x [B, Tin, N]
    Output:
      context_hidden [B, N, H]
      last_hidden_state_flat [B, N*H]  (the GRU hidden state after last step)
      alpha [B, Tin, N]
    """
    def __init__(
        self,
        adj: torch.Tensor,
        hidden_dim: int = 64,
        att_hidden_dim: int = 32,
        att_dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        self.register_buffer("adj", adj)

        self.num_nodes = adj.shape[0]
        self.hidden_dim = hidden_dim

        self.tgcn_cell = TGCNCell(self.adj, input_dim=self.num_nodes, hidden_dim=self.hidden_dim)
        self.attn = TemporalSelfAttention(hidden_dim=self.hidden_dim,
                                          att_hidden_dim=att_hidden_dim,
                                          dropout=att_dropout)

    def forward(self, x: torch.Tensor):
        """
        x: [B, Tin, N]
        """
        B, Tin, N = x.shape
        assert N == self.num_nodes, f"num_nodes mismatch: x has {N}, adj has {self.num_nodes}"

        hidden_state = torch.zeros(B, N * self.hidden_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(Tin):
            out, hidden_state = self.tgcn_cell(x[:, t, :], hidden_state)
            out = out.reshape(B, N, self.hidden_dim)  # [B, N, H]
            outputs.append(out)

        h_seq = torch.stack(outputs, dim=1)  # [B, Tin, N, H]
        context, alpha = self.attn(h_seq)    # context [B, N, H], alpha [B, Tin, N]
        return context, hidden_state, alpha
