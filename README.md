# SMICNN: a convex neural gauge for the 12-D lamination-parameter feasible domain

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the compact research code and frozen constraint model
for learning the feasible domain of the complete 12-dimensional lamination-
parameter vector with a Smooth-Max Input Convex Neural Network (SMICNN).

The released constraint is the **base convex model only**:

- no final output Softplus;
- no threshold-hinge wrapper;
- no positive homogeneous safety head;
- calibration scale fixed to 1 and shift fixed to 0.

For a normalized input vector `x = V / x_scale`, the released model directly
returns the convex gauge approximation `phi(V)`. A deployable inequality is

```text
g_rho(V) = phi(V) - rho <= 0.
```

## Repository layout

```text
SMICNN-12D-LP-Constraint/
├── src/
│   ├── code1_generate_multisource_data.py  # Seven-source StrictLP data
│   ├── code2_train_smicnn.py               # Four-stage SMICNN training
│   ├── code3_validate_metrics.py            # Manuscript evaluation metrics
│   └── checkpoint_io.py                     # Full/compact checkpoint loader
├── models/
│   ├── README.md                            # Model download and usage
│   └── checksums.sha256                     # Release-asset checksum
├── docs/
│   ├── DATA.md                              # Data definition and NPZ schema
│   └── METRICS.md                           # Evaluation definitions/thresholds
├── tools/
│   ├── export_public_checkpoint.py          # Transparent FP32 exporter
│   └── export_quantized_checkpoint.py       # Compact UINT8-storage exporter
├── tests/test_public_checkpoint.py           # Checkpoint integrity smoke test
├── MODEL_CARD.md
├── CITATION.cff
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/LeVingBM/SMICNN-12D-LP-Constraint.git
cd SMICNN-12D-LP-Constraint
python -m pip install -r requirements.txt
```

## Download the frozen model

The recommended `v1.0.1` checkpoint is about 180 MiB. It stores each weight row
with affine UINT8 quantization and reconstructs FP32 weights before inference;
the network architecture and evaluation API are unchanged. It is distributed
as a GitHub Release asset rather than a normal Git object:

```bash
curl -L -o models/smicnn_12d_v1_compact_uint8_storage.pt \
  https://github.com/LeVingBM/SMICNN-12D-LP-Constraint/releases/download/v1.0.1/smicnn_12d_v1_compact_uint8_storage.pt
```

Verify its SHA-256 checksum with `models/checksums.sha256`.

Run the release integrity test after downloading:

```bash
python tests/test_public_checkpoint.py
```

## Quick inference

```python
import torch
from src import load_smicnn_checkpoint

model, x_scale, metadata = load_smicnn_checkpoint(
    "models/smicnn_12d_v1_compact_uint8_storage.pt"
)

V = torch.zeros(1, 12)  # replace with physical lamination parameters
with torch.no_grad():
    phi = model(V / x_scale)

rho = 0.98
g = phi - rho
print("phi =", phi.item(), "feasible =", bool(g.item() <= 0.0))
```

## Generate multi-source StrictLP data

The count order is `dense_random,sparse_random,coordinate_axes,` 
`coordinate_planes_66,random_coordinate_pairs,random_2d_planes,` 
`support_atom_projected`.

```bash
python src/code1_generate_multisource_data.py \
  --output data/train_multisource.npz \
  --counts 1000,300,24,1056,300,720,300 \
  --nz 64 --ntheta 181 --workers 8
```

## Train

```bash
python src/code2_train_smicnn.py \
  --data data/train_multisource.npz \
  --hard-data data/development_hard_slices.npz \
  --output checkpoints/smicnn.pt \
  --device cuda
```

The four stages are deep-ICNN pretraining, adaptive direct supports, hard direct
supports, and hard-mining refinement. The training history is saved next to the
checkpoint with second-resolution timestamps.

## Validate

```bash
python src/code3_validate_metrics.py \
  --checkpoint models/smicnn_12d_v1_compact_uint8_storage.pt \
  --data data/independent_test.npz \
  --output reports/independent_test_metrics.json \
  --rho 0.98 --device cuda
```

The validator reports only the quantities defined in the manuscript: boundary
fitting/classification, fixed and random 2-D slice geometry, deployment safety
with a one-sided 99% Clopper-Pearson upper bound, and gradient quality.

## Reproducibility and scope

The StrictLP labels approximate the continuous lamination-parameter domain by
using `N_z=64` thickness intervals and `N_theta=181` candidate angles. Changing
these values changes the discrete reference domain and therefore requires
regenerating data and revalidating the model.

This release supports reproducible evaluation of the finite sampled protocols;
it is not a mathematical certificate for every point in continuous 12-D space.
For safety-critical optimization, use a conservative `rho` chosen on an
independent distribution and perform an exact StrictLP feasibility check on the
final optimizer output. See [MODEL_CARD.md](MODEL_CARD.md) for known limits.

## Citation

Please use the metadata in [CITATION.cff](CITATION.cff). Update the article DOI
in your manuscript once the paper is published.
