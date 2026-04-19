from __future__ import annotations

import torch
import torch.nn as nn

from .layers import GCNLayer, ResidualMLP


class SeqGraphDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_gcn_layers: int = 2):
        super().__init__()
        self.project = nn.Linear(in_dim, hidden_dim)
        self.gconvs = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)])
        self.residual = ResidualMLP(hidden_dim)
        self.final = nn.Linear(hidden_dim, out_dim)

        nn.init.xavier_uniform_(self.project.weight)
        nn.init.constant_(self.project.bias, 0.0)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.constant_(self.final.bias, 0.0)

    def forward(self, encoded: torch.Tensor, seq_adj: torch.Tensor) -> torch.Tensor:
        h = self.project(encoded)
        for layer in self.gconvs:
            h = layer(h, seq_adj)
        h = self.residual(h)
        return self.final(h)
