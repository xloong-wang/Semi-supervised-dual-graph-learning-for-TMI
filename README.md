# Semi-supervised Dual-Graph Learning for Transportation Mode Identification

This repository contains a cleaned and GitHub-ready implementation of the **B1′ pure-attention dependency graph** variant for GPS-based transportation mode identification using limited-context trajectory segments.

## What is included

- masked graph autoencoder pretraining
- dual-branch encoder with sequence graph and pure-attention dependency graph
- supervised fine-tuning on a small labeled subset
- reproducible train/val/test splits
- command-line entry points for training and evaluation

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── config/
│   └── geolife.yaml
├── data/
│   └── README.md
├── datasets/
│   ├── __init__.py
│   └── geolife.py
├── models/
│   ├── __init__.py
│   ├── layers.py
│   ├── decoder.py
│   └── mgae.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── seed.py
│   └── splits.py
├── scripts/
│   ├── train_geolife.sh
│   └── eval_geolife.sh
├── checkpoints/
└── results/
```

## Expected data layout

Place the processed GeoLife `.npy` files under a directory such as:

```text
/data/my_dataset_npy/
├── feat.npy
├── adj.npy
├── labels.npy
├── diff_feat.npy      # optional
└── diff_adj.npy       # optional
```

The loader expects:

- `feat.npy`: shape `[S, N, F]`
- `adj.npy`: shape `[S, N, N]`
- `labels.npy`: shape `[S]`
- `diff_feat.npy` and `diff_adj.npy` are optional placeholders

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data_dir /path/to/my_dataset_npy --config config/geolife.yaml
```

## Evaluation

```bash
python evaluate.py   --data_dir /path/to/my_dataset_npy   --config config/geolife.yaml   --checkpoint checkpoints/b1prime_best.pt
```

## Notes on the split protocol

The default split uses:

- 60% train
- 20% validation
- 20% test

Within the 60% training split, the default fine-tuning subset uses **5% of the training split**, and the remaining training samples are used for self-supervised pretraining. This matches the behavior of the original experiment script after cleanup.

## Citation

If you use this code, please cite your paper here.
