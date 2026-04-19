# Semi-supervised Dual-graph Learning for Transportation Mode Identification

## Overview

This repository provides the implementation of a semi-supervised dual-graph learning framework for transportation mode identification. Each trajectory segment is modeled with two graph views:

- a **Sequence Graph** for local temporal continuity
- a **Dependency Graph** for non-local dependency modeling

To improve representation learning under limited labeled data, the framework further incorporates a **masked graph autoencoder (MGAE)** pretraining strategy.

The current release focuses on the main **B1′ pure-attention dependency graph** setting.

## Repository Structure

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

## Environment

The code is implemented in Python and PyTorch.

Recommended environment:

- Python 3.10
- PyTorch 2.8.0
- CUDA 12.8

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The processed GeoLife dataset can be downloaded from:

- Baidu Netdisk: `https://pan.baidu.com/s/1pTKGpiVAwxy7ndfM4FXOhw?pwd=nn23`
- Extraction code: `nn23`

After downloading and extracting `my_dataset_npy.7z`, please place the folder under:

```text
data/my_dataset_npy/
```

The processed dataset directory should contain:

```text
feat.npy
adj.npy
labels.npy
```

### Data Format

- `feat.npy`: node features with shape `[S, N, F]`
- `adj.npy`: sequence graph adjacency matrices with shape `[S, N, N]`
- `labels.npy`: segment labels with shape `[S]`

where:

- `S` denotes the number of trajectory segments
- `N` denotes the number of GPS points in each segment
- `F` denotes the feature dimension

## Training

To train the model, run:

```bash
python train.py --data_dir ./data/my_dataset_npy --config config/geolife.yaml
```

or use the provided script:

```bash
bash scripts/train_geolife.sh
```

## Evaluation

To evaluate a trained checkpoint, run:

```bash
python evaluate.py --data_dir ./data/my_dataset_npy --config config/geolife.yaml --ckpt ./checkpoints/b1prime_best.pt
```

or use:

```bash
bash scripts/eval_geolife.sh
```

## Experimental Setting

By default, the dataset is split into:

- 60% training
- 20% validation
- 20% test

Within the training set, a supervised subset is used for fine-tuning, while the remaining training samples are used for self-supervised pretraining.

Please refer to `utils/splits.py` and `config/geolife.yaml` for the detailed settings.

## Notes

- The current release focuses on the main setting.
- The dependency graph is dynamically constructed from raw features through attention-based adjacency learning.
- The code assumes that the processed data have already been generated and saved in `.npy` format.
- Minor differences in results may still occur across different hardware and software environments.

## Reproducibility

Random seeds are fixed in the code to improve reproducibility.
Please note that slight performance variations may still appear due to differences in hardware, CUDA versions, and library implementations.
