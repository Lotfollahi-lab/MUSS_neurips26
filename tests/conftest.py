"""Tiny synthetic corpus in the processed layout (written with the real preprocessing writer)."""

import os
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muss.preprocess.common import (  # noqa: E402
    GEXDatasetWriter, chrom_id_map, peaks_meta, split_positions, split_unseen_forced,
)

D_INPUT = 16
N_CELLS = 24
CELLTYPES = ["A", "B", "C"]


def _peaks() -> pd.DataFrame:
    rows = [("chr1", 10_000 + 6_000 * k, 10_500 + 6_000 * k) for k in range(40)]
    rows += [("chr2", 5_000 + 9_000 * k, 5_800 + 9_000 * k) for k in range(20)]
    return pd.DataFrame(rows, columns=["Chromosome", "Start", "End"])


def _genes():
    chrom = ["chr1", "chr1", "chr1", "chr1", "chr2", "chr2", "chrX"]
    tss = [20_000, 60_000, 150_000, 200_000, 30_000, 120_000, 1_000]
    ids = [f"ENSG{i:011d}" for i in range(len(tss))]
    id_map = chrom_id_map(chrom)
    return {"gene_ids": ids, "gene_symbols": [f"G{i}" for i in range(len(tss))], "n_genes": len(tss), "tss": tss,
            "Chromosome": chrom, "Chromosome_id": [id_map[c[3:]] for c in chrom]}


def _write_h5(path, dset, n, rng, ids=None):
    with h5py.File(path, "w") as h5:
        h5.create_dataset(dset, data=rng.normal(size=(n, D_INPUT)).astype(np.float32))
        if ids is not None:
            h5.create_dataset("gene_ids", data=np.array([i.encode() for i in ids], dtype="S"))


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """Paths of processed PBMC datasets and a sequence-fold BED."""
    root = tmp_path_factory.mktemp("processed")
    rng = np.random.RandomState(0)
    peaks, genes = _peaks(), _genes()
    pm = peaks_meta(peaks)

    # PBMC10K-like GEX dataset
    out = os.path.join(root, "pbmc10k")
    cell_ids = [f"cell{i}" for i in range(N_CELLS)]
    celltypes = [CELLTYPES[i % 3] for i in range(N_CELLS)]
    splits = split_unseen_forced(cell_ids, celltypes, 0.2, 0.2, ["C"], seed=0)
    writer = GEXDatasetWriter(out, split_positions(splits, cell_ids), max_chunk_size_mb=1)
    for i in range(N_CELLS):
        pids = np.unique(rng.choice(len(peaks), size=rng.randint(3, 25), replace=False)).astype(np.int32)
        gex = rng.poisson(2.0, size=genes["n_genes"]).astype(np.float32)
        writer.add(i, gex, pids, celltypes[i], cell_ids[i])
    writer.finalize(genes, pm, {"dataset": "tiny"})
    _write_h5(os.path.join(out, "gene_embeddings_alphagenome.h5"), "gene_embeddings", genes["n_genes"], rng, genes["gene_ids"])

    # shared peak embeddings
    os.makedirs(os.path.join(root, "shared"))
    _write_h5(os.path.join(root, "shared", "peak_embeddings_alphagenome.h5"), "peak_embeddings", len(peaks), rng)

    # PBMC169K-like scATAC batches
    b_root = os.path.join(root, "pbmc169k")
    os.makedirs(os.path.join(b_root, "batches"))
    batches, g = {}, 0
    for name, n in (("B1", 10), ("B2", 8)):
        recs = []
        for i in range(n):
            pids = np.unique(rng.choice(len(peaks), size=rng.randint(3, 25), replace=False)).tolist()
            split = ["train", "train", "valid", "test"][g % 4]
            recs.append({"idx": i, "data": {"peak_ids": pids, "meta": {"celltype": "NA", "global_idx": g, "split": split}}})
            g += 1
        with open(os.path.join(b_root, "batches", f"{name}.pkl"), "wb") as f:
            pickle.dump(recs, f)
        batches[name] = {"batch_name": name, "output_pkl": f"batches/{name}.pkl", "n_cells": n}
    with open(os.path.join(b_root, "overview.pkl"), "wb") as f:
        pickle.dump({"peaks": pm, "batches": batches}, f)

    # sequence folds: chr1 [0, 131072) and chr2 [131072, 262144) are held out
    bed = os.path.join(root, "sequences.bed")
    with open(bed, "w") as f:
        for chrom in ("chr1", "chr2", "chrX"):
            for k in range(3):
                fold = "fold3" if (chrom, k) in {("chr1", 0), ("chr2", 1)} else "fold0"
                f.write(f"{chrom}\t{k * 131072}\t{(k + 1) * 131072}\t{fold}\n")
        f.write("chr3\t0\t131072\tfold4\n")  # the second default held-out fold (no peaks or genes there)
    return {"root": str(root), "pbmc10k": out, "bed": bed, "n_peaks": len(peaks), "n_genes": genes["n_genes"]}
