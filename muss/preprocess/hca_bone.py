"""HCA developing bone multiome (48 samples) -> processed GEX dataset.

    python -m muss.preprocess.hca_bone --raw RAW --reference REF --pbmc10k OUT/processed/pbmc/pbmc10k \
        --out OUT/processed/hca_bone --weights W

* cells: ATAC ``passed_qc`` and an annotated cell type; samples in directory-name order
* genes: the PBMC10K gene vocabulary (genes not measured are zero)
* peaks: ``consensus_peaks.bed`` on chr1-22, X, Y, M, sorted by (chromosome name, start, end)
* split: per cell type 80/10/10 (cell types sorted by name), held-out cell types only in
  valid/test (seed 42)

Inputs: ``RAW/hca_bone/consensus_peaks.bed`` and ``RAW/hca_bone/<sample>/<sample>_sharedbarcodes_{gex,atac}.h5ad``
(shipped in ``MUSS_assets/raw``): pycisTopic consensus peaks and per-sample raw GEX + ATAC counts
computed with cellgeni/nf-atac from the data of To et al., Nature 2024 (ArrayExpress E-MTAB-14385).
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import (
    GEXDatasetWriter, canonical_chrom, check_fasta_has_canonical_chroms, chrom_key, copy_reference_gene_embeddings,
    dense_gene_index, load_reference_genes, log, peaks_meta, split_positions, split_stratified, strip_version,
    write_json,
)

N_SAMPLES = 48
UNSEEN_CELLTYPES = ["ChondroPro2", "MandibularChon", "SutureMes1", "Osteoblast", "LymphaticEndothelial", "Schwann",
                    "Erythroid", "Arachnoid", "PerineuralFIB"]


def sample_files(root: str, sample: str) -> Tuple[str, str]:
    return (os.path.join(root, sample, f"{sample}_sharedbarcodes_gex.h5ad"),
            os.path.join(root, sample, f"{sample}_sharedbarcodes_atac.h5ad"))


def discover_samples(root: str) -> List[str]:
    """The sample directories (sorted); every one must hold both h5ad files, and all 48 must be present."""
    samples = [s for s in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, s))]
    incomplete = [s for s in samples if not all(os.path.exists(f) for f in sample_files(root, s))]
    if incomplete:
        raise FileNotFoundError(f"{len(incomplete)} sample directories under {root} lack an h5ad file, e.g. {incomplete[:3]}")
    if len(samples) != N_SAMPLES:
        raise FileNotFoundError(f"expected {N_SAMPLES} sample directories under {root}, found {len(samples)}")
    return samples


def consensus_peaks(bed: str) -> pd.DataFrame:
    df = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 1, 2], names=["Chromosome", "Start", "End"],
                     dtype={"Chromosome": str, "Start": np.int64, "End": np.int64})
    df["Chromosome"] = [canonical_chrom(c) for c in df["Chromosome"]]
    df = df[df["Chromosome"].notna()].copy()
    df["_key"] = df["Chromosome"].map(chrom_key)
    df = df.sort_values(["_key", "Start", "End"], kind="mergesort").drop_duplicates(subset=["Chromosome", "Start", "End"])
    return df[["Chromosome", "Start", "End"]].reset_index(drop=True)


def _obs_index(obs: h5py.Group) -> np.ndarray:
    key = obs.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else str(key)
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in obs[key][...]])


def _obs_column(obs: h5py.Group, col: str) -> np.ndarray:
    node = obs[col]
    if isinstance(node, h5py.Group):  # categorical
        cats = [x.decode() if isinstance(x, bytes) else str(x) for x in node["categories"][...]]
        return np.array([None if c < 0 else cats[c] for c in node["codes"][...]], dtype=object)
    arr = node[...]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr]) if arr.dtype.kind in "SO" else arr


def kept_cells(root: str, samples: List[str]) -> Dict[str, List[Tuple[int, str, str]]]:
    """sample -> [(row, cell id, cell type)] of the cells passing ATAC QC with a cell type."""
    out = {}
    for s in samples:
        gex_path, atac_path = sample_files(root, s)
        with h5py.File(gex_path, "r") as g, h5py.File(atac_path, "r") as a:
            ids = _obs_index(g["obs"])
            if not np.array_equal(ids, _obs_index(a["obs"])):
                raise RuntimeError(f"{s}: GEX and ATAC cells differ")
            qc = np.array([bool(x) for x in _obs_column(a["obs"], "passed_qc")], dtype=bool)
            ct = _obs_column(g["obs"], "celltype")
            has_ct = np.array([not (x is None or (isinstance(x, float) and np.isnan(x))) for x in ct], dtype=bool)
            out[s] = [(int(r), str(ids[r]), str(ct[r])) for r in np.flatnonzero(qc & has_ct)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="raw data root (contains hca_bone/)")
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
    root = os.path.join(a.raw, "hca_bone")

    samples = discover_samples(root)
    vocab = consensus_peaks(os.path.join(root, "consensus_peaks.bed"))
    key_to_id = {k: i for i, k in enumerate(vocab.itertuples(index=False, name=None))}
    log(f"[peaks] {len(vocab)} peaks, {len(samples)} samples")

    genes = load_reference_genes(a.pbmc10k)
    first = ad.read_h5ad(sample_files(root, samples[0])[0], backed="r")
    gid_col = "gene_ids" if "gene_ids" in first.var.columns else "gene_id"
    dst, src = dense_gene_index([strip_version(g) for g in first.var[gid_col].astype(str)], genes["gene_ids"])
    first.file.close()
    log(f"[genes] {len(dst)}/{genes['n_genes']} vocabulary genes measured")

    cells = kept_cells(root, samples)
    cell_ids = [c for s in samples for _, c, _ in cells[s]]
    celltypes = [t for s in samples for _, _, t in cells[s]]
    log(f"[cells] {len(cell_ids)} cells")
    splits = split_stratified(cell_ids, celltypes, 0.1, 0.1, UNSEEN_CELLTYPES, 42)
    write_json(splits, os.path.join(a.out, "split_id.json"))

    writer = GEXDatasetWriter(a.out, split_positions(splits, cell_ids))
    g = 0
    for s in samples:
        gex_path, atac_path = sample_files(root, s)
        gex, atac = ad.read_h5ad(gex_path), ad.read_h5ad(atac_path)
        if not gex.obs.index.equals(atac.obs.index):
            raise RuntimeError(f"{s}: GEX and ATAC cells differ")
        var = atac.var
        l2g = np.array([key_to_id.get((canonical_chrom(c), int(st), int(en)), -1) for c, st, en in
                        zip(var["Chromosome"].astype(str), var["Start"].astype(np.int64), var["End"].astype(np.int64))],
                       dtype=np.int32)
        X = gex.X.tocsr() if sp.issparse(gex.X) else sp.csr_matrix(np.asarray(gex.X))
        A = atac.X.tocsr() if sp.issparse(atac.X) else sp.csr_matrix(np.asarray(atac.X))
        ct = gex.obs["celltype"].astype("string")
        bc = gex.obs["barcode"].astype(str) if "barcode" in gex.obs.columns else pd.Series(gex.obs.index.astype(str))
        for r, _, _ in cells[s]:
            gx = np.zeros(genes["n_genes"], dtype=np.float32)
            gx[dst] = X.getrow(r).toarray().ravel().astype(np.float32, copy=False)[src]
            row = A.getrow(r)
            ids = l2g[row.indices[row.data > 0].astype(np.int64, copy=False)]
            ids = np.unique(ids[ids >= 0]).astype(np.int32)
            writer.add(g, gx, ids, "NA" if pd.isna(ct.iat[r]) else str(ct.iat[r]), str(bc.iat[r]))
            g += 1
        log(f"[cells] {s}: {len(cells[s])} cells (total {g})")
        del gex, atac
    writer.finalize(genes, peaks_meta(vocab), {"dataset": "hca_bone", "peak_vocab": "consensus_peaks_sorted",
                                               "samples": samples, "unseen_celltypes": UNSEEN_CELLTYPES})

    if not a.skip_embeddings:
        from .alphagenome import peak_embeddings

        peak_embeddings(vocab, fasta, a.weights, os.path.join(a.out, "peak_embeddings_alphagenome.h5"))
        copy_reference_gene_embeddings(a.pbmc10k, a.out, genes["gene_ids"])
    log(f"[done] {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
