from __future__ import annotations

from pathlib import Path
import numpy as np


def _ensure_diff_placeholders(num_samples: int, feature_dim: int, n_dep: int = 18):
    diff_feat = np.zeros((num_samples, n_dep, feature_dim), dtype=np.float32)
    diff_adj = np.zeros((num_samples, n_dep, n_dep), dtype=np.float32)
    return diff_adj, diff_feat


def load_geolife_npy(data_dir: str | Path):
    """Load processed GeoLife arrays from a directory.

    Expected files:
    - feat.npy:   [S, N, F]
    - adj.npy:    [S, N, N]
    - labels.npy: [S]

    Optional files:
    - diff_feat.npy
    - diff_adj.npy
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    feat_fp = data_dir / 'feat.npy'
    adj_fp = data_dir / 'adj.npy'
    labels_fp = data_dir / 'labels.npy'

    missing = [p.name for p in [feat_fp, adj_fp, labels_fp] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {data_dir}: {missing}")

    feat = np.load(feat_fp).astype(np.float32)
    adj = np.load(adj_fp).astype(np.float32)
    labels = np.load(labels_fp).astype(np.int64)

    if feat.ndim != 3:
        raise ValueError(f"feat.npy must have shape [S, N, F], got {feat.shape}")
    if adj.ndim != 3:
        raise ValueError(f"adj.npy must have shape [S, N, N], got {adj.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels.npy must have shape [S], got {labels.shape}")

    num_samples, num_nodes, feature_dim = feat.shape
    if adj.shape[0] != num_samples or labels.shape[0] != num_samples:
        raise ValueError('Sample dimension mismatch among feat, adj, and labels')
    if adj.shape[1] != num_nodes or adj.shape[2] != num_nodes:
        raise ValueError('Node dimension mismatch between feat and adj')

    diff_feat_fp = data_dir / 'diff_feat.npy'
    diff_adj_fp = data_dir / 'diff_adj.npy'
    if diff_feat_fp.exists() and diff_adj_fp.exists():
        diff_feat = np.load(diff_feat_fp).astype(np.float32)
        diff_adj = np.load(diff_adj_fp).astype(np.float32)
    else:
        diff_adj, diff_feat = _ensure_diff_placeholders(num_samples, feature_dim)

    return {
        'feat': feat,
        'adj': adj,
        'labels': labels,
        'diff_feat': diff_feat,
        'diff_adj': diff_adj,
    }
