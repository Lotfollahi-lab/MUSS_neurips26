# Training settings

[Overview](../README.md) · [Reproduction commands](reproduction.md)

## Fine-tuning

All encoder and prediction-head parameters are fine-tuned. AlphaGenome embeddings remain
fixed. Training uses fp32, seed 1, one cell per device per forward pass and no gradient clipping.

| Dataset | Reference GPUs | Cells per device per update | Effective batch size | Budget |
|---|---|---|---|---|
| PBMC10K | 4 | 2 | 8 | 80 epochs / 56,560 updates |
| BMMC | 2 | 4 | 8 | 24,000 updates |
| Open Problems | 2 | 4 | 8 | 14,592 updates |
| HCA bone | 4 | 4 | 16 | 24,000 updates |

The CLI selects these defaults from `configs/datasets.json`. `--devices 1` adjusts accumulation
to retain the effective batch size. Set `--accumulate` to choose it explicitly. Changing the
device count changes sample assignment and dropout draws; resume a run with its original configuration.

## Optimizer and loss

The Poisson NLL is summed over seen-sequence genes and averaged over cells. Unseen-sequence
genes are excluded from the model during training and included during evaluation.

`MUSSAdam` retains the original Adam update: betas `(0.9, 0.98)`, epsilon `1e-6` before
second-moment bias correction, and no weight decay. The learning rate starts at zero,
warms up to `1e-4` over 100 optimizer updates, then follows inverse-square-root decay.
Lightning manages accumulation and distributed training; sample ordering, final-group
gradient normalization and update-based random seeds follow the released training configuration.

## Data and splits

Preprocessing uses seed 42. Cell types held out of training are recorded in
`chunks/overview.pkl`. Genes and peaks in sequence folds `fold3` and `fold4` are excluded
from training. Each cell uses at most 10,000 accessible peaks in genomic order.

Open Problems uses normalized, log1p-transformed expression targets; the other datasets
use raw counts. Dataset splits and sequence embeddings are prepared locally by the supplied scripts.

## Hardware

The released models were fine-tuned on NVIDIA H200 GPUs (141 GB). The unchanged model and
fp32 computation can require approximately 95 GB of GPU memory for fine-tuning and 40 GB
for evaluation, depending on the cell's peak count. A single GPU must still fit one cell;
reducing device count does not reduce this per-device requirement.
