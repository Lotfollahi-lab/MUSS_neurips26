"""NeurIPS 2021 BMMC multiome (GEO GSE194122) -> processed GEX dataset.

    python -m muss.preprocess.bmmc --raw RAW --reference REF --pbmc10k OUT/processed/pbmc/pbmc10k \
        --out OUT/processed/bmmc --weights W

* cells: GEX_n_counts > 0 and ATAC_nCount_peaks > 0; raw counts from ``layers['counts']``
* genes: the PBMC10K gene vocabulary (genes BMMC does not measure are zero)
* peaks: BMMC's peaks on chr1-22, X, Y, M, sorted by (chromosome name, start, end)
* split: 80/10/10, cells of the held-out cell types only in valid/test (seed 42)

Input: ``RAW/bmmc/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad`` (the gunzipped
GEO supplementary file of GSE194122; see the README for the download command).
"""

import argparse
import os
import re
import time
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import (
    GEXDatasetWriter, canonical_chrom, check_fasta_has_canonical_chroms, chrom_key, copy_reference_gene_embeddings,
    dense_gene_index, load_reference_genes, log, peaks_meta, split_positions, split_unseen_forced,
    strip_version, write_json,
)

H5AD = "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
UNSEEN_CELLTYPES = ["Erythroblast", "Proerythroblast", "Normoblast", "CD16+ Mono", "CD8+ T naive", "pDC",
                    "ID2-hi myeloid prog", "Plasma cell", "ILC"]
_PEAK_NAME = [re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$"),
              re.compile(r"^(?P<chrom>[^-]+)-(?P<start>\d+)-(?P<end>\d+)$")]


def feature_masks(var: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    ft = var["feature_types"].astype(str).str.strip().str.lower()
    genes = ft.isin({"gex", "gene expression", "rna", "expression"}) | ft.str.contains("gene", regex=False)
    peaks = ft.isin({"atac", "peak", "peaks", "chromatin accessibility"}) | ft.str.contains("atac|peak|chromatin", regex=True)
    return genes.to_numpy(), peaks.to_numpy()


def parse_peak_name(name: str) -> Tuple[str, int, int]:
    """'chr1-9776-10668' or 'chr1:9776-10668' -> ('chr1', 9776, 10668)."""
    for rx in _PEAK_NAME:
        m = rx.match(str(name).strip())
        if m:
            s, e = int(m.group("start")), int(m.group("end"))
            return m.group("chrom"), min(s, e), max(s, e)
    raise ValueError(f"cannot parse peak name {name!r}")


def local_peak_vocabulary(names) -> Tuple[pd.DataFrame, np.ndarray]:
    """(sorted unique canonical peaks, vocabulary id of every feature or -1)."""
    rows = []
    for pos, name in enumerate(names):
        chrom, s, e = parse_peak_name(name)
        c = canonical_chrom(chrom)
        if c is not None:
            rows.append((pos, c, s, e))
    df = pd.DataFrame(rows, columns=["pos", "Chromosome", "Start", "End"])
    df["_key"] = df["Chromosome"].map(chrom_key)
    df = df.sort_values(["_key", "Start", "End", "pos"], kind="mergesort")
    vocab = df.drop_duplicates(subset=["Chromosome", "Start", "End"], keep="first")[["Chromosome", "Start", "End"]]
    vocab = vocab.reset_index(drop=True)
    key_to_id = {k: i for i, k in enumerate(vocab.itertuples(index=False, name=None))}
    ids = np.full(len(names), -1, dtype=np.int32)
    for pos, c, s, e in rows:
        ids[pos] = key_to_id[(c, s, e)]
    return vocab, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="raw data root (contains bmmc/)")
    ap.add_argument("--reference", required=True, help="reference directory (genome.fa)")
    ap.add_argument("--pbmc10k", required=True, help="processed PBMC10K dataset (gene vocabulary + gene embeddings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", help="AlphaGenome model_all_folds.safetensors (required unless --skip-embeddings)")
    ap.add_argument("--skip-embeddings", action="store_true", help="write no AlphaGenome embedding files")
    a = ap.parse_args()
    if not a.skip_embeddings and not a.weights:
        ap.error("--weights is required unless --skip-embeddings is given")
    t0 = time.time()
    fasta = os.path.join(a.reference, "genome.fa")
    check_fasta_has_canonical_chroms(fasta)
    os.makedirs(a.out, exist_ok=True)

    adata = ad.read_h5ad(os.path.join(a.raw, "bmmc", H5AD), backed="r")
    matrix = adata.layers["counts"]
    obs, var = adata.obs.copy(), adata.var.copy()
    is_gene, is_peak = feature_masks(var)
    gene_cols, peak_cols = np.flatnonzero(is_gene), np.flatnonzero(is_peak)
    gid_col = "gene_id" if "gene_id" in var.columns else "gene_ids"
    source_gene_ids = [strip_version(g) for g in var.loc[is_gene, gid_col].astype(str)]

    genes = load_reference_genes(a.pbmc10k)
    dst, src = dense_gene_index(source_gene_ids, genes["gene_ids"])
    log(f"[genes] {len(dst)}/{genes['n_genes']} vocabulary genes measured")

    vocab, peak_id = local_peak_vocabulary(var.index[is_peak].astype(str).tolist())
    log(f"[peaks] {len(vocab)} peaks ({int(is_peak.sum())} features)")

    keep = np.ones(len(obs), dtype=bool)
    for col in ("GEX_n_counts", "ATAC_nCount_peaks"):
        keep &= pd.to_numeric(obs[col], errors="coerce").fillna(0.0).to_numpy() > 0.0
    rows = np.flatnonzero(keep)
    cell_ids = obs.index.astype(str).to_numpy()[rows].tolist()
    celltypes = obs["cell_type"].astype(str).iloc[rows].tolist()
    splits = split_unseen_forced(cell_ids, celltypes, 0.1, 0.1, UNSEEN_CELLTYPES, 42)
    write_json(splits, os.path.join(a.out, "split_id.json"))

    writer = GEXDatasetWriter(a.out, split_positions(splits, cell_ids))
    for i, r in enumerate(rows.tolist()):
        g = matrix[r, gene_cols]
        g = np.asarray(g.toarray() if hasattr(g, "toarray") else g).reshape(-1).astype(np.float32, copy=False)
        gx = np.zeros(genes["n_genes"], dtype=np.float32)
        gx[dst] = g[src]
        p = matrix[r, peak_cols]
        active = p.tocsr().indices if sp.issparse(p) else np.flatnonzero(np.asarray(p).reshape(-1) > 0)
        ids = peak_id[active.astype(np.int32, copy=False)]
        ids = np.unique(ids[ids >= 0]).astype(np.int32)
        writer.add(i, gx, ids, celltypes[i], cell_ids[i])
        if (i + 1) % 10000 == 0:
            log(f"[cells] {i + 1}/{len(rows)}")
    writer.finalize(genes, peaks_meta(vocab), {"dataset": "bmmc", "source": H5AD, "matrix": "layers['counts']",
                                               "peak_vocab": "local_sorted", "unseen_celltypes": UNSEEN_CELLTYPES})
    adata.file.close()

    if not a.skip_embeddings:
        from .alphagenome import peak_embeddings

        peak_embeddings(vocab, fasta, a.weights, os.path.join(a.out, "peak_embeddings_alphagenome.h5"))
        copy_reference_gene_embeddings(a.pbmc10k, a.out, genes["gene_ids"])
    log(f"[done] {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
