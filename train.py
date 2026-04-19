from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from datasets import load_geolife_npy
from models import MaskedGAE
from utils.metrics import accuracy_from_logits
from utils.seed import set_seed
from utils.splits import make_splits


def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class Classifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, nclass: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, nclass)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, h: torch.Tensor):
        z = h.sum(dim=1)
        z = F.relu(self.fc1(z))
        return self.fc2(z)


def iterate_minibatches(indices: np.ndarray, batch_size: int, shuffle: bool = True):
    indices = indices.copy()
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/geolife.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg['save_dir'])
    result_dir = Path(cfg['result_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    data = load_geolife_npy(args.data_dir)
    feat = data['feat']
    adj = data['adj']
    labels = data['labels']

    num_samples, num_nodes, feature_dim = feat.shape
    nclass = int(labels.max() + 1)

    splits = make_splits(
        num_samples=num_samples,
        seed=cfg['seed'],
        train_ratio=cfg['split']['train_ratio'],
        val_ratio=cfg['split']['val_ratio'],
        test_ratio=cfg['split']['test_ratio'],
        finetune_ratio_within_train=cfg['split']['finetune_ratio_within_train'],
        exclude_finetune_from_pretrain=cfg['split']['exclude_finetune_from_pretrain'],
    )

    feat_t = torch.from_numpy(feat).float().to(device)
    adj_t = torch.from_numpy(adj).float().to(device)
    labels_t = torch.from_numpy(labels).long().to(device)
    feat_dep_t = feat_t.clone()

    model = MaskedGAE(
        nfeat=feature_dim,
        nhid=cfg['model']['hidden_dim'],
        recon_dim=feature_dim,
        num_nodes=num_nodes,
        d_att=cfg['model']['attention_dim'],
        tau=cfg['model']['tau'],
        topk=cfg['model']['topk'],
    ).to(device)

    optimizer_pre = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    mse = nn.MSELoss()

    print(f'Device: {device}')
    print(f'Dataset shape: feat={feat.shape}, adj={adj.shape}, labels={labels.shape}, nclass={nclass}')
    print(f"Splits: pretrain={len(splits['pretrain_idx'])}, finetune={len(splits['finetune_idx'])}, val={len(splits['val_idx'])}, test={len(splits['test_idx'])}")

    # ---------- pretraining ----------
    for epoch in range(cfg['training']['pretrain_epochs']):
        model.train()
        loss_history = []
        for batch_idx in iterate_minibatches(splits['pretrain_idx'], cfg['training']['batch_size']):
            x_hat, mask_idx, _, a_dep = model.forward_pretrain(
                feat_t[batch_idx],
                adj_t[batch_idx],
                feat_dep_t[batch_idx],
                cfg['pretrain']['mask_ratio'],
                cfg['pretrain']['remask_ratio'],
            )
            if mask_idx.sum() == 0:
                continue
            mask_broadcast = mask_idx.unsqueeze(-1).expand_as(x_hat)
            recon_loss = mse(x_hat[mask_broadcast], feat_t[batch_idx][mask_broadcast])
            entropy = -(a_dep.clamp_min(1e-9) * torch.log(a_dep.clamp_min(1e-9))).sum(-1).mean()
            loss = recon_loss + cfg['pretrain']['entropy_reg'] * entropy

            optimizer_pre.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_norm'])
            optimizer_pre.step()
            loss_history.append(loss.item())

        mean_loss = float(np.mean(loss_history)) if loss_history else float('nan')
        print(f"[Pretrain] Epoch {epoch + 1}/{cfg['training']['pretrain_epochs']} loss={mean_loss:.6f}")

    torch.save({'model': model.state_dict(), 'config': cfg}, save_dir / 'b1prime_pretrained.pt')

    # ---------- fine-tuning ----------
    classifier = Classifier(
        in_dim=2 * cfg['model']['hidden_dim'],
        hidden_dim=cfg['model']['classifier_hidden_dim'],
        nclass=nclass,
    ).to(device)

    if cfg['training']['freeze_encoder']:
        for param in model.parameters():
            param.requires_grad = False

    params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(classifier.parameters())
    optimizer_ft = torch.optim.Adam(params, lr=cfg['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = 0.0

    for epoch in range(cfg['training']['finetune_epochs']):
        model.train()
        classifier.train()
        loss_history = []
        for batch_idx in iterate_minibatches(splits['finetune_idx'], cfg['training']['batch_size']):
            h, _, _, _ = model.encode_nodes(feat_t[batch_idx], adj_t[batch_idx], feat_dep_t[batch_idx])
            logits = classifier(h)
            loss = criterion(logits, labels_t[batch_idx])

            optimizer_ft.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg['training']['clip_norm'])
            optimizer_ft.step()
            loss_history.append(loss.item())

        model.eval()
        classifier.eval()
        with torch.no_grad():
            h_val, _, _, _ = model.encode_nodes(feat_t[splits['val_idx']], adj_t[splits['val_idx']], feat_dep_t[splits['val_idx']])
            logits_val = classifier(h_val)
            val_acc = accuracy_from_logits(logits_val, labels_t[splits['val_idx']])

        mean_loss = float(np.mean(loss_history)) if loss_history else float('nan')
        print(f"[Finetune] Epoch {epoch + 1}/{cfg['training']['finetune_epochs']} train_loss={mean_loss:.6f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'config': cfg,
                'val_acc': val_acc,
                'epoch': epoch + 1,
                'splits': {k: v.tolist() for k, v in splits.items()},
            }
            torch.save(best_state, save_dir / 'b1prime_best.pt')

    if best_state is None:
        raise RuntimeError('No best model was saved during fine-tuning.')

    model.load_state_dict(best_state['model'])
    classifier.load_state_dict(best_state['classifier'])
    model.eval()
    classifier.eval()
    with torch.no_grad():
        h_test, _, _, _ = model.encode_nodes(feat_t[splits['test_idx']], adj_t[splits['test_idx']], feat_dep_t[splits['test_idx']])
        logits_test = classifier(h_test)
        test_acc = accuracy_from_logits(logits_test, labels_t[splits['test_idx']])

    print(f'[Test] accuracy={test_acc:.4f} (best_val_acc={best_val_acc:.4f})')

    summary = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'num_samples': num_samples,
        'num_nodes': num_nodes,
        'feature_dim': feature_dim,
        'nclass': nclass,
        'split_sizes': {k: int(len(v)) for k, v in splits.items()},
    }
    save_json(summary, result_dir / 'summary.json')


if __name__ == '__main__':
    main()
