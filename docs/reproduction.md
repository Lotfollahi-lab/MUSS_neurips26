# Reproduction guide

[Overview](../README.md) · [Training settings](settings.md) ·
[Metric definitions](metrics.md) · [Reference results](results.md)

## Environment setup

Run the commands from the cloned code directory in the same Bash session:

```bash
git clone https://github.com/Lotfollahi-lab/MUSS_neurips26.git
cd MUSS_neurips26
conda env create -f environment.yml
conda activate muss
pip install -e .
set -o pipefail
mkdir -p logs
```

The environment pins Python 3.10, PyTorch 2.8.0 and CUDA 12.8, with Lightning 2.5.6.
See [hardware requirements](settings.md#hardware) before running training.

### Download data and checkpoints

Download both Hugging Face repositories into the same `MUSS_ASSETS` directory, preserving
their folder structure:

| Repository | Paths inside `MUSS_ASSETS` |
|---|---|
| [kangjiezheng/MUSS-data](https://huggingface.co/datasets/kangjiezheng/MUSS-data) (dataset) | `raw/pbmc10k/`, `raw/pbmc169k/`, `raw/hca_bone/`, `reference/` |
| [kangjiezheng/MUSS](https://huggingface.co/kangjiezheng/MUSS) (model) | `checkpoints/pretrain/`, `checkpoints/finetune/` |

If you already have these files, set `MUSS_ASSETS` to their absolute path and skip the downloads.
`MUSS_DATA` is the separate output directory for locally preprocessed data.

```bash
export MUSS_ASSETS="$PWD/../MUSS_assets"
hf download kangjiezheng/MUSS-data --repo-type dataset --local-dir "$MUSS_ASSETS"
hf download kangjiezheng/MUSS --repo-type model --include "checkpoints/*" --local-dir "$MUSS_ASSETS"
export MUSS_DATA="$PWD/data/processed"
```

To omit HCA bone, append `--exclude "raw/hca_bone/*"` to the data download.

## Data preprocessing

Preprocess every dataset you will use, including when evaluating released checkpoints.
The scripts start from expression/accessibility matrices and annotations, not sequencing reads.
Build PBMC first: the other datasets reuse its gene panel and gene embeddings.

### Input datasets

| Dataset | Original data source | Input for the MUSS preprocessing scripts |
|---|---|---|
| PBMC10K | [10x Genomics](https://www.10xgenomics.com/datasets) | Supplied `raw/pbmc10k/`: paired ATAC/GEX `.h5ad` files and SEACells assignments |
| PBMC169K | De Rop et al., [GEO GSE194028](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194028) | Supplied `raw/pbmc169k/`: `.h5ad` files for 47 batches |
| HCA bone | To et al., [ArrayExpress E-MTAB-14385](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-14385), [developmental atlas downloads](https://developmental.cellatlas.io/skeleton-development) | Supplied `raw/hca_bone/`: paired ATAC/GEX `.h5ad` files for 48 samples and consensus peaks |
| BMMC | Luecken et al., [GEO GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) | Download the processed multiome `.h5ad` below |
| Open Problems 2022 | [Kaggle competition data](https://www.kaggle.com/competitions/open-problems-multimodal/data) | Download `metadata.csv`, `train_multi_inputs.h5` and `train_multi_targets.h5` below |

For HCA bone, use the supplied `raw/hca_bone/` inputs: the original study's processed download
has different peaks, expression processing and annotations.

### Download BMMC and Open Problems inputs

Skip this step for PBMC/HCA-only work. Kaggle downloads require an
[API token](https://www.kaggle.com/docs/api) and acceptance of the competition rules.

```bash
REPO="$PWD"
RAW="$MUSS_ASSETS/raw"
(
    set -e
    mkdir -p "$RAW/bmmc" "$RAW/openproblems"
    cd "$RAW/bmmc"
    wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
    gzip -dkf GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
    cd "$RAW/openproblems"
    for f in metadata.csv train_multi_inputs.h5 train_multi_targets.h5; do
        kaggle competitions download -c open-problems-multimodal -f "$f"
    done
    for archive in *.zip; do
        if [ -f "$archive" ]; then unzip -o "$archive"; fi
    done
    cd "$RAW"
    sha256sum -c "$REPO/configs/raw_checksums.sha256"
)
```

### Build the datasets

Install the pinned AlphaGenome PyTorch port and download its weights:

```bash
pip install --no-deps --ignore-requires-python \
    "git+https://github.com/genomicsxai/alphagenome-pytorch@6e96bb5"
hf download gtca/alphagenome_pytorch model_all_folds.safetensors --local-dir weights/
W="$PWD/weights/model_all_folds.safetensors"
```

Run PBMC first, then keep only the datasets you need in the loop below. For PBMC-only work,
omit the loop. Embedding computation uses one GPU, with a slower CPU fallback.

```bash
RAW="$MUSS_ASSETS/raw"
REF="$MUSS_ASSETS/reference"
mkdir -p "$MUSS_DATA" logs/preprocess

python -m muss.preprocess.pbmc --raw "$RAW" --reference "$REF" \
    --out "$MUSS_DATA/pbmc" --weights "$W" \
    2>&1 | tee logs/preprocess/pbmc.log

for dataset in bmmc openproblems hca_bone; do
    python -m "muss.preprocess.$dataset" --raw "$RAW" --reference "$REF" \
        --pbmc10k "$MUSS_DATA/pbmc/pbmc10k" --out "$MUSS_DATA/$dataset" \
        --weights "$W" 2>&1 | tee "logs/preprocess/$dataset.log"
done
```

The scripts use `reference/hvg_genes.csv`, the 5,257-gene HVG list selected for MUSS, and
write the following data directories:

| Task/dataset | Processed data directory |
|---|---|
| PBMC clustering | `$MUSS_DATA/pbmc` |
| PBMC10K gene-expression prediction | `$MUSS_DATA/pbmc/pbmc10k` |
| BMMC | `$MUSS_DATA/bmmc` |
| Open Problems | `$MUSS_DATA/openproblems` |
| HCA bone | `$MUSS_DATA/hca_bone` |

## Fine-tuning

The released encoder is `$MUSS_ASSETS/checkpoints/pretrain/muss_pretrain.pt`.
After preprocessing, fine-tune it on a downstream dataset:

```bash
python scripts/finetune.py --dataset bmmc --data "$MUSS_DATA/bmmc" \
    --out-dir runs/finetune_bmmc --log-dir logs/finetune \
    2>&1 | tee logs/finetune_bmmc.log
```

Replace `bmmc` with `pbmc10k`, `openproblems` or `hca_bone` and set `--data` using the
[processed directory table](#build-the-datasets). Each dataset has its own defaults in
[training settings](settings.md). `MUSS_ASSETS` supplies the encoder checkpoint and sequence-fold BED;
use `--pretrained` and `--sequences-bed` to override those paths.

The default uses the reference device count. To use one GPU, add `--devices 1`; gradient
accumulation is adjusted to preserve the effective batch size. Hardware changes can affect
numerical results. Training uses fp32 and one cell per device per forward pass.

Each run writes:

- `muss_finetuned.pt`: portable weights for evaluation.
- `last.ckpt`: model, optimizer, scheduler and data position for resuming training; saved every 4,000 updates, at epoch boundaries and at the final update.
- CSV training metrics under `--log-dir`.

Use a new output directory for an independent run. To resume, keep the data, seed, device
count and accumulation unchanged:

```bash
python scripts/finetune.py --dataset bmmc --data "$MUSS_DATA/bmmc" \
    --out-dir runs/finetune_bmmc --log-dir logs/finetune \
    --resume runs/finetune_bmmc/last.ckpt \
    2>&1 | tee logs/finetune_bmmc_resume.log
```

For evaluation without fine-tuning, the model repository provides:

| Dataset | Checkpoint relative to `MUSS_ASSETS` |
|---|---|
| PBMC10K | `checkpoints/finetune/muss_pbmc10k.pt` |
| BMMC | `checkpoints/finetune/muss_bmmc.pt` |
| Open Problems | `checkpoints/finetune/muss_openproblems.pt` |
| HCA bone | `checkpoints/finetune/muss_hca_bone.pt` |

## Evaluation

Expression evaluation accepts released weights, `muss_finetuned.pt` or `last.ckpt`.
Both evaluation scripts use one GPU, with a CPU fallback.

### Gene-expression prediction

Evaluate the released fine-tuned models on the test split; keep only the datasets you need
in the loop:

```bash
mkdir -p logs/evaluation
for dataset in pbmc10k bmmc openproblems hca_bone; do
    data="$MUSS_DATA/$dataset"
    if [ "$dataset" = pbmc10k ]; then data="$MUSS_DATA/pbmc/pbmc10k"; fi
    python scripts/evaluate_gex.py --dataset "$dataset" --data "$data" \
        --checkpoint "$MUSS_ASSETS/checkpoints/finetune/muss_$dataset.pt" \
        --out-dir "results/eval_$dataset" \
        2>&1 | tee "logs/evaluation/$dataset.log"
done
```

For your own model, change `--checkpoint` and `--out-dir`, for example:

```bash
python scripts/evaluate_gex.py --dataset bmmc --data "$MUSS_DATA/bmmc" \
    --checkpoint runs/finetune_bmmc/muss_finetuned.pt \
    --out-dir results/eval_bmmc_local 2>&1 | tee logs/eval_bmmc_local.log
```

Each run writes `main_table.csv`, `metrics_summary.csv` and `metrics_per_celltype.csv`.
Add `--save-predictions` to save `predictions.npz`.

### Zero-shot cell clustering

Evaluate the released encoder on the two held-out PBMC169K batches:

```bash
python scripts/cell_clustering.py --data "$MUSS_DATA/pbmc" \
    --checkpoint "$MUSS_ASSETS/checkpoints/pretrain/muss_pretrain.pt" \
    --out-dir results/clustering 2>&1 | tee logs/clustering.log
```

Results are saved in `clustering_summary.csv`, alongside per-batch embeddings, labels and
per-seed metrics. See [metric definitions](metrics.md) and [reference results](results.md).

## Checks

Run the CPU tests, including checkpoint loading, loss, sampling and resume checks:

```bash
pytest -q
```
