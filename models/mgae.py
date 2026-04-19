from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from .decoder import SeqGraphDecoder
from .layers import GCN


class DynamicAdjacencyBuilder(nn.Module):
    """Pure-attention dependency graph based on raw node features."""

    def __init__(self, in_dim: int, d_att: int = 32, tau: float = 0.7, topk: int = 12):
        super().__init__()
        self.q = nn.Linear(in_dim, d_att, bias=False)
        self.k = nn.Linear(in_dim, d_att, bias=False)
        self.tau = tau
        self.topk = topk
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)

    @staticmethod
    def _row_topk(a: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or k >= a.size(-1):
            return a
        _, idx = torch.topk(a, k=k, dim=-1)
        mask = torch.zeros_like(a).scatter_(-1, idx, 1.0)
        a_sparse = a * mask
        return a_sparse / (a_sparse.sum(-1, keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor):
        q = self.q(x)
        k = self.k(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        a = torch.softmax(scores / self.tau, dim=-1)
        a = self._row_topk(a, self.topk)
        return a, scores


def mask_input_features(x: torch.Tensor, mask_ratio: float, mask_token: torch.Tensor):
    batch_size, num_nodes, _ = x.shape
    x_masked = x.clone()
    mask_idx = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=x.device)
    num_mask = max(1, int(math.ceil(mask_ratio * num_nodes)))
    for i in range(batch_size):
        chosen = np.random.choice(num_nodes, num_mask, replace=False)
        mask_idx[i, chosen] = True
        x_masked[i, chosen, :] = mask_token[0, 0, :]
    return x_masked, mask_idx


def remask_node_embeddings(h: torch.Tensor, remask_ratio: float, dmask_token: torch.Tensor):
    batch_size, num_nodes, _ = h.shape
    h_remask = h.clone()
    remask_idx = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=h.device)
    num_mask = max(1, int(math.ceil(remask_ratio * num_nodes)))
    for i in range(batch_size):
        chosen = np.random.choice(num_nodes, num_mask, replace=False)
        remask_idx[i, chosen] = True
        h_remask[i, chosen, :] = dmask_token[0, 0, :]
    return h_remask, remask_idx


class MaskedGAE(nn.Module):
    def __init__(self, nfeat: int, nhid: int, recon_dim: int, num_nodes: int, d_att: int, tau: float, topk: int):
        super().__init__()
        self.encoder_seq = GCN(nfeat, nhid)
        self.encoder_dep = GCN(nfeat, nhid)
        self.decoder = SeqGraphDecoder(in_dim=2 * nhid, hidden_dim=nhid, out_dim=recon_dim, num_gcn_layers=2)
        self.num_nodes = num_nodes
        self.mask_token = nn.Parameter(torch.randn(1, 1, nfeat) * 0.02)
        self.dmask_token = nn.Parameter(torch.randn(1, 1, 2 * nhid) * 0.02)
        self.adj_dyn = DynamicAdjacencyBuilder(in_dim=nfeat, d_att=d_att, tau=tau, topk=topk)

    def forward_pretrain(self, feat_seq: torch.Tensor, adj_seq: torch.Tensor, feat_dep_seq: torch.Tensor, mask_ratio: float, remask_ratio: float):
        _, num_nodes, _ = feat_seq.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f'num_nodes mismatch: got {num_nodes}, expected {self.num_nodes}')

        feat_masked, mask_idx = mask_input_features(feat_seq, mask_ratio, self.mask_token)
        node_s, _ = self.encoder_seq(feat_masked, adj_seq)
        a_dep, _ = self.adj_dyn(feat_dep_seq)
        node_d, _ = self.encoder_dep(feat_dep_seq, a_dep)

        h = torch.cat([node_s, node_d], dim=-1)
        h_remask, remask_idx = remask_node_embeddings(h, remask_ratio, self.dmask_token)
        x_hat = self.decoder(h_remask, adj_seq)
        return x_hat, mask_idx, remask_idx, a_dep

    def encode_nodes(self, feat_seq: torch.Tensor, adj_seq: torch.Tensor, feat_dep_seq: torch.Tensor):
        node_s, graph_s = self.encoder_seq(feat_seq, adj_seq)
        a_dep, _ = self.adj_dyn(feat_dep_seq)
        node_d, graph_d = self.encoder_dep(feat_dep_seq, a_dep)
        h = torch.cat([node_s, node_d], dim=-1)
        return h, graph_s, graph_d, a_dep
