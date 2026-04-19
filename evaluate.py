from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from datasets import load_geolife_npy
from models import MaskedGAE
from train import Classifier
from utils.metrics import accuracy_from_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/geolife.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = load_geolife_npy(args.data_dir)
    feat = torch.from_numpy(data['feat']).float().to(device)
    adj = torch.from_numpy(data['adj']).float().to(device)
    labels = torch.from_numpy(data['labels']).long().to(device)
    feat_dep = feat.clone()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    splits = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in checkpoint['splits'].items()}

    num_samples, num_nodes, feature_dim = data['feat'].shape
    nclass = int(data['labels'].max() + 1)

    model = MaskedGAE(
        nfeat=feature_dim,
        nhid=cfg['model']['hidden_dim'],
        recon_dim=feature_dim,
        num_nodes=num_nodes,
        d_att=cfg['model']['attention_dim'],
        tau=cfg['model']['tau'],
        topk=cfg['model']['topk'],
    ).to(device)
    classifier = Classifier(2 * cfg['model']['hidden_dim'], cfg['model']['classifier_hidden_dim'], nclass).to(device)

    model.load_state_dict(checkpoint['model'])
    classifier.load_state_dict(checkpoint['classifier'])
    model.eval()
    classifier.eval()

    with torch.no_grad():
        h_test, _, _, _ = model.encode_nodes(feat[splits['test_idx']], adj[splits['test_idx']], feat_dep[splits['test_idx']])
        logits = classifier(h_test)
        test_acc = accuracy_from_logits(logits, labels[splits['test_idx']])

    summary = {
        'checkpoint': args.checkpoint,
        'test_acc': test_acc,
        'num_samples': int(num_samples),
        'num_nodes': int(num_nodes),
        'feature_dim': int(feature_dim),
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
