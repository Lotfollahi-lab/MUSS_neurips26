#!/usr/bin/env python3
"""Zero-shot cell clustering on the two held-out PBMC169K batches.

    python scripts/cell_clustering.py --data "$MUSS_DATA/pbmc" \
        --checkpoint $MUSS_ASSETS/checkpoints/pretrain/muss_pretrain.pt --out-dir results/clustering

Run PBMC preprocessing first; --data points to the locally generated corpus.

A cell embedding is the mean of the released encoder's output over the cell's peaks
(no fine-tuning, no masking). For every batch: PCA to 50 components (scanpy), then k-means
with k = number of annotated cell types for 10 seeds; ARI / FMI / NMI / AMI against the
annotation are reported as mean +- std (ddof=0) over the seeds. PCA and k-means run on one
thread, which makes the scores reproducible (multi-threaded k-means is not bit-deterministic).
"""

import argparse
import json
import logging
import os
import sys
import time
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from muss.checkpoint import load_checkpoint_state, model_config_from_checkpoint  # noqa: E402
from muss.data.common import PeakDictionary, gene_meta, load_embeddings, load_pickle, peak_meta  # noqa: E402
from muss.models import MUSSEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)
logger = logging.getLogger("cell_clustering")

BATCHES = ["VIB_10xv1_1", "BIO_ddseq_1"]
KMEANS_SEEDS = [23, 42, 7, 11, 99, 17, 31, 8, 44, 100]
METRICS = {"ARI": adjusted_rand_score, "FMI": fowlkes_mallows_score, "NMI": normalized_mutual_info_score, "AMI": adjusted_mutual_info_score}


@torch.no_grad()
def embed_batch(model, task, batch_file: str, max_peaks: int, device) -> tuple:
    peak_start = torch.as_tensor(task.peak_meta["Start"], dtype=torch.long)
    peak_end = torch.as_tensor(task.peak_meta["End"], dtype=torch.long)
    peak_chr = torch.as_tensor(task.peak_meta["Chromosome_id"].astype(np.int64))
    gene_tss = torch.as_tensor(task.gene_meta["tss"], dtype=torch.long).to(device)
    gene_chr = torch.as_tensor(task.gene_meta["Chromosome_id"].astype(np.int64)).to(device)
    embs, labels, skipped = [], [], 0
    records = load_pickle(batch_file)
    t0 = time.time()
    for i, sample in enumerate(records):
        rec = sample["data"]
        peak_ids = torch.as_tensor(np.asarray(rec["peak_ids"], dtype=np.int64)[:max_peaks])
        if peak_ids.numel() == 0:
            skipped += 1
            continue
        x, _ = model.encode(
            src_tokens=(peak_ids + task.dictionary.n_special)[None].to(device),
            peak_padding_mask=torch.zeros((1, peak_ids.numel()), dtype=torch.bool, device=device),
            peak_chrom_id=peak_chr[peak_ids][None].to(device),
            peak_start=peak_start[peak_ids][None].to(device),
            peak_end=peak_end[peak_ids][None].to(device),
            gene_chrom_id=gene_chr,
            gene_tss=gene_tss,
        )
        enc = x.float()
        mask = torch.ones((1, enc.shape[1], 1), dtype=enc.dtype, device=device)
        embs.append(((enc * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).cpu().numpy())
        labels.append(str((rec.get("meta") or {}).get("celltype", "unknown")))
        if i % 1000 == 0:
            logger.info("%s: %d/%d cells (%.0fs)", os.path.basename(batch_file), i, len(records), time.time() - t0)
    if skipped:
        logger.warning("%d cells without peaks were skipped", skipped)
    return np.concatenate(embs).astype(np.float32), np.asarray(labels, dtype=object)


def cluster(emb: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    adata = ad.AnnData(X=emb.astype(np.float32), obs=pd.DataFrame({"celltype": labels}, index=[str(i) for i in range(len(labels))]))
    k = int(pd.Series(labels).nunique())
    rows = []
    with threadpool_limits(limits=1):  # multi-threaded k-means can converge differently from run to run
        sc.tl.pca(adata, n_comps=int(min(50, max(2, emb.shape[0] - 1), emb.shape[1])))
        for seed in KMEANS_SEEDS:
            pred = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(adata.obsm["X_pca"])
            rows.append({"seed": seed, **{m: float(f(labels, pred)) for m, f in METRICS.items()}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="released MUSS encoder checkpoint")
    ap.add_argument("--assets", default=os.environ.get("MUSS_ASSETS"))
    ap.add_argument("--data", help="processed PBMC corpus (default: <assets>/processed/pbmc)")
    ap.add_argument("--batches", default=",".join(BATCHES))
    ap.add_argument("--max-cell-peaks-num", type=int, default=10000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if not args.data and not args.assets:
        ap.error("pass --data or --assets (or set MUSS_ASSETS)")
    data = args.data or os.path.join(args.assets, "processed", "pbmc")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overview = load_pickle(os.path.join(data, "pbmc10k", "chunks", "overview.pkl"))
    pm, gm = peak_meta(overview), gene_meta(overview)
    task = SimpleNamespace(peak_meta=pm, gene_meta=gm, dictionary=PeakDictionary(pm["n_peaks"]))
    peaks = load_embeddings(os.path.join(data, "shared", "peak_embeddings_alphagenome.h5"), "peak_embeddings")
    genes = load_embeddings(os.path.join(data, "pbmc10k", "gene_embeddings_alphagenome.h5"), "gene_embeddings")
    if len(peaks) != pm["n_peaks"] or len(genes) != gm["n_genes"]:
        raise ValueError("Embedding rows do not match the dataset metadata")
    state = load_checkpoint_state(args.checkpoint)
    model = MUSSEncoder(model_config_from_checkpoint(state), task.dictionary,
                        torch.from_numpy(peaks), torch.from_numpy(genes))
    model.load_pretrained_encoder(state)
    model.to(device).eval()

    ov169 = load_pickle(os.path.join(data, "pbmc169k", "overview.pkl"))
    summary = []
    for batch in [b for b in args.batches.split(",") if b]:
        path = os.path.join(data, "pbmc169k", "batches", os.path.basename(str(ov169["batches"][batch]["output_pkl"])))
        emb, labels = embed_batch(model, task, path, args.max_cell_peaks_num, device)
        np.save(os.path.join(args.out_dir, f"{batch}__embeddings.npy"), emb)
        pd.DataFrame({"celltype": labels}).to_csv(os.path.join(args.out_dir, f"{batch}__labels.csv"), index=False)
        runs = cluster(emb, labels)
        runs.to_csv(os.path.join(args.out_dir, f"{batch}__kmeans_metrics.csv"), index=False)
        row = {"batch": batch, "n_cells": len(labels), "n_celltypes": int(pd.Series(labels).nunique())}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = float(runs[m].mean()), float(runs[m].std(ddof=0))
        summary.append(row)
        logger.info("%s: %s", batch, json.dumps(row))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out_dir, "clustering_summary.csv"), index=False)
    if torch.cuda.is_available():
        logger.info("peak GPU memory allocated: %.1f GiB", torch.cuda.max_memory_allocated() / 2**30)
    print("\nPBMC169K zero-shot cell clustering (mean +- std over 10 k-means seeds):")
    for _, r in summary.iterrows():
        print(f"  {r['batch']:<12} " + "  ".join(f"{m}={r[f'{m}_mean']:.4f}+-{r[f'{m}_std']:.4f}" for m in METRICS))


if __name__ == "__main__":
    main()
