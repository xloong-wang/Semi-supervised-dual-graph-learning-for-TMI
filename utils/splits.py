from __future__ import annotations

import numpy as np


def make_splits(
    num_samples: int,
    seed: int = 227,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    finetune_ratio_within_train: float = 0.05,
    exclude_finetune_from_pretrain: bool = True,
):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError('train_ratio + val_ratio + test_ratio must sum to 1.')

    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_samples)

    s1 = int(train_ratio * num_samples)
    s2 = int((train_ratio + val_ratio) * num_samples)

    train_idx = perm[:s1]
    val_idx = perm[s1:s2]
    test_idx = perm[s2:]

    num_finetune = max(1, int(finetune_ratio_within_train * len(train_idx)))
    finetune_idx = train_idx[:num_finetune]

    if exclude_finetune_from_pretrain:
        pretrain_idx = train_idx[num_finetune:]
    else:
        pretrain_idx = train_idx.copy()

    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'pretrain_idx': pretrain_idx,
        'finetune_idx': finetune_idx,
    }
