from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.PReLU()
        nn.init.xavier_uniform_(self.fc.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.fc(feat)
        out = torch.bmm(adj, h)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


class GCN(nn.Module):
    def __init__(self, nfeat: int, nhid: int):
        super().__init__()
        self.g1 = GCNLayer(nfeat, nhid)
        self.g2 = GCNLayer(nhid, nhid)
        self.g3 = GCNLayer(nhid, nhid)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        x = F.relu(self.g1(x, adj))
        x = F.relu(self.g2(x, adj))
        x = self.g3(x, adj)
        node_emb = x
        graph_emb = torch.sum(x, dim=1)
        return node_emb, graph_emb


class ResidualMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.PReLU(),
            nn.Linear(dim, dim),
            nn.PReLU(),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) + x
