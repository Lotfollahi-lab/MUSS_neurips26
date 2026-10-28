"""PBMC10K and PBMC169K preprocessing with one shared peak vocabulary.

    python -m muss.preprocess.pbmc --raw RAW --reference REF --out OUT/processed/pbmc --weights W

1. PBMC10K ATAC is smoothed with its SEACells metacells: a cell keeps a peak if it has a read
   there and the peak is open (mean > 0.05) across the cell's metacell. Cells outside metacells
   are dropped.
2. Peak vocabulary: the PBMC10K peaks (sorted) followed by the PBMC169K peaks that match no
   PBMC10K peak; a match needs both boundaries less than 50 bp apart (closest match wins).
3. Gene vocabulary: the reference highly variable genes (by symbol) that have a gene interval,
   in PBMC10K order (5,257 genes); it is reused by every fine-tuning dataset.
4. PBMC10K: 80/10/10 split, cells of the held-out cell types only in valid/test (seed 42).
5. PBMC169K: random 80/10/10 split over all cells (seed 42); one pickle per batch.
6. AlphaGenome embeddings of all peaks (``shared/``) and genes (``pbmc10k/``).

Inputs (``RAW``): ``pbmc10k/pbmc10k_coupled_{atac,gex}.h5ad``,
``pbmc10k/seacell_metacell_assignments.csv``, ``pbmc169k/<batch>/<batch>.h5ad``.
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import (
    GEXDatasetWriter, canonical_chrom, check_fasta_has_canonical_chroms, chrom_id_map, genes_meta, load_gene_intervals,
    log, peaks_meta, select_gene_intervals, split_positions, split_random, split_unseen_forced, write_json,
    write_pickle,
)

UNSEEN_CELLTYPES = ["TEM_TEMRA_CYTOTOXIC_T_CELLS", "REGULATORY_T_CELLS", "CLASSICAL_MONOCYTES",
                    "NON-CLASSICAL_MONOCYTES", "DC2", "PDC"]
METACELL_THRESHOLD = 0.05
MATCH_BP = 50
SEED = 42
N_BATCHES = 47  # PBMC169K


# ----------------------------------------------------------------------------
# peaks
# ----------------------------------------------------------------------------
def peak_table(var: pd.DataFrame) -> pd.DataFrame:
    """Chromosome / Start / End of every feature (row order kept), plus a canonical-chromosome flag."""
    chrom = [canonical_chrom(c) for c in var["Chromosome"]]
    start = pd.to_numeric(var["Start"], errors="coerce").astype("Int64").astype("int64").to_numpy()
    end = pd.to_numeric(var["End"], errors="coerce").astype("Int64").astype("int64").to_numpy()
    start, end = np.minimum(start, end), np.maximum(start, end)
    return pd.DataFrame({"Chromosome": [c or "" for c in chrom], "Start": start, "End": end,
                         "canonical": [c is not None for c in chrom]})


def unique_peaks(var: pd.DataFrame) -> pd.DataFrame:
    df = peak_table(var)
    return df.loc[df["canonical"], ["Chromosome", "Start", "End"]].drop_duplicates(keep="first").reset_index(drop=True)


def local_to_global(var: pd.DataFrame, key_to_id: Dict[Tuple[str, int, int], int]) -> np.ndarray:
    """Vocabulary id of every feature column (-1 for non-canonical chromosomes)."""
    df = peak_table(var)
    out = np.full(len(df), -1, dtype=np.int64)
    ok = df["canonical"].to_numpy()
    keys = zip(df["Chromosome"][ok], df["Start"][ok].astype(int), df["End"][ok].astype(int))
    mapped = np.array([key_to_id.get(k, -1) for k in keys], dtype=np.int64)
    if (mapped < 0).any():
        raise RuntimeError(f"{int((mapped < 0).sum())} canonical peaks are missing from the vocabulary")
    out[np.flatnonzero(ok)] = mapped
    return out


def match_peaks(secondary: pd.DataFrame, primary: pd.DataFrame, delta: int = MATCH_BP) -> np.ndarray:
    """For each secondary peak the primary peak with |d_start| < delta and |d_end| < delta
    minimizing |d_start| + |d_end| (ties: smaller id), else -1."""
    p_start, p_end = primary["Start"].to_numpy(np.int64), primary["End"].to_numpy(np.int64)
    buckets: Dict[str, Dict[Tuple[int, int], List[int]]] = {}
    for i, (c, s, e) in enumerate(zip(primary["Chromosome"].astype(str), p_start, p_end)):
        buckets.setdefault(c, {}).setdefault((int(s) // delta, int(e) // delta), []).append(i)
    out = np.full(len(secondary), -1, dtype=np.int64)
    for i, (c, s, e) in enumerate(zip(secondary["Chromosome"].astype(str), secondary["Start"].to_numpy(np.int64),
                                      secondary["End"].to_numpy(np.int64))):
        cb = buckets.get(c)
        if cb is None:
            continue
        s, e = int(s), int(e)
        best, best_d = -1, None
        for bx in (s // delta - 1, s // delta, s // delta + 1):
            for by in (e // delta - 1, e // delta, e // delta + 1):
                for pid in cb.get((bx, by), ()):
                    ds, de = abs(s - int(p_start[pid])), abs(e - int(p_end[pid]))
                    if ds >= delta or de >= delta:
                        continue
                    d = ds + de
                    if best < 0 or d < best_d or (d == best_d and pid < best):
                        best, best_d = int(pid), int(d)
        out[i] = best
    return out


def shared_vocabulary(pbmc10k: pd.DataFrame, pbmc169k: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """(vocabulary, vocabulary id of every PBMC169K unique peak)."""
    matched = match_peaks(pbmc169k, pbmc10k)
    extra = pbmc169k.loc[matched < 0].reset_index(drop=True)
    vocab = pd.concat([pbmc10k, extra], ignore_index=True)[["Chromosome", "Start", "End"]]
    ids = matched.copy()
    ids[matched < 0] = np.arange(len(pbmc10k), len(vocab), dtype=np.int64)
    log(f"[peaks] PBMC10K {len(pbmc10k)} | PBMC169K {len(pbmc169k)} ({int((matched >= 0).sum())} matched) "
        f"| vocabulary {len(vocab)}")
    return vocab, ids


# ----------------------------------------------------------------------------
# PBMC10K
# ----------------------------------------------------------------------------
def metacell_atac(atac_h5ad: str, assignments_csv: str, threshold: float = METACELL_THRESHOLD) -> ad.AnnData:
    """Binary ATAC of the cells in a metacell: read in the cell AND metacell mean > threshold."""
    adata = ad.read_h5ad(atac_h5ad)
    mc = pd.read_csv(assignments_csv)
    if "barcode" not in adata.obs.columns:
        adata.obs["barcode"] = adata.obs_names.astype(str)
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    labels = mc["metacell"].astype(str)
    rows_all, blocks = [], []
    for name in labels.unique().tolist():
        barcodes = {str(b).rsplit("___", 1)[0] for b in mc.loc[labels == name, "barcode"]}
        rows = np.flatnonzero(adata.obs["barcode"].isin(barcodes).to_numpy())
        x = X[rows].toarray()
        keep = x.mean(axis=0) > threshold
        rows_all.append(rows)
        blocks.append(sp.csr_matrix(np.logical_and(x > 0, keep).astype(np.int8)))
    rows_all = np.concatenate(rows_all)
    order = np.argsort(rows_all, kind="mergesort")
    out = ad.AnnData(X=sp.vstack(blocks, format="csr")[order], obs=adata.obs.iloc[rows_all[order]].copy(),
                     var=adata.var.copy())
    log(f"[pbmc10k] metacell-smoothed ATAC: {out.n_obs}/{adata.n_obs} cells in {labels.nunique()} metacells")
    return out


def gene_id_col(var: pd.DataFrame) -> str:
    for c in ("gene_ids", "gene_id"):
        if c in var.columns:
            return c
    raise KeyError("GEX .var has no gene_ids / gene_id column")


def hvg_genes(gex_var: pd.DataFrame, hvg_csv: str, interval_genes: set) -> Tuple[List[str], List[str]]:
    """Reference HVGs (matched by symbol) with a gene interval, in GEX order: (gene ids, symbols)."""
    wanted = set(pd.read_csv(hvg_csv)["gene_name"].astype(str))
    symbols = gex_var.index.astype(str)
    ids = gex_var[gene_id_col(gex_var)].astype(str).to_numpy()
    keep = np.asarray(symbols.isin(wanted)) & np.array([g in interval_genes for g in ids], dtype=bool)
    return ids[keep].tolist(), symbols.to_numpy()[keep].tolist()


def _core_barcode(x) -> str:
    return str(x).split("___", 1)[0]


def shared_names(values) -> List[str]:
    """``values`` as str, with one object per distinct value: pickle then stores each name once
    per chunk file, as in the published PBMC10K chunks."""
    names: Dict[str, str] = {}
    return [names.setdefault(v, v) for v in map(str, values)]


def build_pbmc10k(atac: ad.AnnData, gex_h5ad: str, gene_ids: List[str], gene_symbols: List[str],
                  intervals: pd.DataFrame, vocab: pd.DataFrame, key_to_id, out_dir: str) -> None:
    gex = ad.read_h5ad(gex_h5ad)
    l2g = local_to_global(atac.var, key_to_id)
    pos = {g: i for i, g in enumerate(gex.var[gene_id_col(gex.var)].astype(str))}
    gex = gex[:, np.asarray([pos[g] for g in gene_ids], dtype=np.int64)].copy()

    gene_chroms = intervals["Chromosome"].astype(str).tolist()
    id_map = chrom_id_map(gene_chroms + vocab["Chromosome"].astype(str).tolist())
    genes = genes_meta(gene_ids, gene_symbols, intervals, id_map)
    peaks = peaks_meta(vocab, id_map)

    for a in (gex, atac):
        if "barcode" not in a.obs.columns:
            a.obs["barcode"] = a.obs_names.astype(str)
        a.obs["barcode"] = a.obs["barcode"].map(_core_barcode)
        a.obs_names = a.obs["barcode"].astype(str)
        a.obs_names_make_unique()
    common = gex.obs_names[gex.obs_names.isin(atac.obs_names)]
    gex, atac = gex[common, :].copy(), atac[common, :].copy()
    cell_ids = gex.obs["barcode"].astype(str).tolist()
    celltypes = shared_names(gex.obs["celltype"])
    log(f"[pbmc10k] {len(cell_ids)} cells with GEX and ATAC")

    splits = split_unseen_forced(cell_ids, celltypes, 0.1, 0.1, UNSEEN_CELLTYPES, SEED)
    write_json(splits, os.path.join(out_dir, "split_id.json"))
    writer = GEXDatasetWriter(out_dir, split_positions(splits, cell_ids))
    X = gex.X.tocsr() if sp.issparse(gex.X) else gex.X
    A = atac.X.tocsr() if sp.issparse(atac.X) else sp.csr_matrix(atac.X)
    for i in range(len(cell_ids)):
        gx = X[i].toarray()[0] if sp.issparse(X) else np.asarray(X[i]).ravel()
        ids = l2g[A[i].indices.astype(np.int32)]
        ids = np.unique(ids[ids >= 0]).astype(np.int32)
        writer.add(i, gx.astype(np.float32, copy=False), ids, celltypes[i], cell_ids[i])
    writer.finalize(genes, peaks, {"dataset": "pbmc10k", "peak_vocab": "shared_pbmc10k_pbmc169k",
                                   "metacell_threshold": METACELL_THRESHOLD, "unseen_celltypes": UNSEEN_CELLTYPES,
                                   "split_seed": SEED})


# ----------------------------------------------------------------------------
# PBMC169K
# ----------------------------------------------------------------------------
def pbmc169k_batches(root: str) -> List[Tuple[str, str]]:
    """(name, h5ad) of the batch directories (sorted); every one must hold ``<batch>.h5ad``, and all
    47 must be present."""
    names = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    incomplete = [d for d in names if not os.path.isfile(os.path.join(root, d, f"{d}.h5ad"))]
    if incomplete:
        raise FileNotFoundError(f"{len(incomplete)} batch directories under {root} lack <batch>.h5ad, e.g. {incomplete[:3]}")
    if len(names) != N_BATCHES:
        raise FileNotFoundError(f"expected {N_BATCHES} batch directories under {root}, found {len(names)}")
    return [(d, os.path.join(root, d, f"{d}.h5ad")) for d in names]


def build_pbmc169k(batches: List[Tuple[str, str]], vocab: pd.DataFrame, key_to_id, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "batches"), exist_ok=True)
    n_cells = []
    for _, path in batches:
        a = ad.read_h5ad(path, backed="r")
        n_cells.append(int(a.n_obs))
        a.file.close()
    splits = {k[:-4]: v for k, v in split_random(sum(n_cells), 0.1, 0.1, SEED).items()}
    write_json(splits, os.path.join(out_dir, "split.json"))
    split_of = {int(p): name for name, pos in splits.items() for p in pos}

    summaries, g = {}, 0
    for (name, path), n in zip(batches, n_cells):
        atac = ad.read_h5ad(path)
        l2g = local_to_global(atac.var, key_to_id)
        obs = atac.obs
        barcode = obs["barcode"].astype(str) if "barcode" in obs.columns else pd.Series(obs.index.astype(str))
        sample_id = obs["sample_id"].astype(str) if "sample_id" in obs.columns else pd.Series([name] * n)
        celltype = obs["celltype"].astype(str) if "celltype" in obs.columns else pd.Series(["NA"] * n)
        A = atac.X.tocsr() if sp.issparse(atac.X) else sp.csr_matrix(atac.X)
        records, counts, g0 = [], {"train": 0, "valid": 0, "test": 0}, g
        for i in range(n):
            ids = l2g[A[i].indices.astype(np.int32)]
            ids = np.unique(ids[ids >= 0]).astype(np.int32)
            split = split_of[g]
            counts[split] += 1
            meta = {"celltype": str(celltype.iat[i]), "barcode": str(barcode.iat[i]), "sample_id": str(sample_id.iat[i]),
                    "batch_key": name, "obs_name": str(obs.index[i]), "global_idx": g, "split": split}
            records.append({"idx": i, "data": {"peak_ids": ids.tolist(), "meta": meta}})
            g += 1
        write_pickle(records, os.path.join(out_dir, "batches", f"{name}.pkl"))
        summaries[name] = {"batch_name": name, "output_pkl": f"batches/{name}.pkl", "n_cells": n,
                           "n_local_peaks": int(atac.n_vars), "global_idx_range": [g0, g], "split_counts": counts}
        log(f"[pbmc169k] {name}: {n} cells")
    write_pickle({"format": "muss-processed-v1", "peaks": peaks_meta(vocab), "splits": splits, "batches": summaries,
                  "dataset_meta": {"dataset": "pbmc169k", "modality": "ATAC-only", "n_cells_total": g,
                                   "peak_vocab": "shared_pbmc10k_pbmc169k", "split_seed": SEED}},
                 os.path.join(out_dir, "overview.pkl"))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="raw data root (contains pbmc10k/ and pbmc169k/)")
    ap.add_argument("--reference", required=True, help="reference directory (genome.fa, gene_intervals.csv, hvg_genes.csv)")
    ap.add_argument("--out", required=True, help="output directory (processed/pbmc)")
    ap.add_argument("--weights", help="AlphaGenome model_all_folds.safetensors (required unless --skip-embeddings)")
    ap.add_argument("--skip-embeddings", action="store_true", help="write no AlphaGenome embedding files")
    a = ap.parse_args()
    if not a.skip_embeddings and not a.weights:
        ap.error("--weights is required unless --skip-embeddings is given")
    t0 = time.time()
    fasta = os.path.join(a.reference, "genome.fa")
    check_fasta_has_canonical_chroms(fasta)
    raw10k = os.path.join(a.raw, "pbmc10k")
    out10k, out169k, shared = (os.path.join(a.out, d) for d in ("pbmc10k", "pbmc169k", "shared"))
    for d in (out10k, out169k, shared):
        os.makedirs(d, exist_ok=True)

    atac = metacell_atac(os.path.join(raw10k, "pbmc10k_coupled_atac.h5ad"),
                         os.path.join(raw10k, "seacell_metacell_assignments.csv"))
    batches = pbmc169k_batches(os.path.join(a.raw, "pbmc169k"))
    p10k = unique_peaks(atac.var).sort_values(["Chromosome", "Start", "End"]).reset_index(drop=True)
    p169k = pd.concat([unique_peaks(ad.read_h5ad(p, backed="r").var) for _, p in batches], ignore_index=True)
    p169k = p169k.drop_duplicates(keep="first").sort_values(["Chromosome", "Start", "End"]).reset_index(drop=True)
    vocab, p169k_ids = shared_vocabulary(p10k, p169k)
    key_to_id = {(c, int(s), int(e)): i for i, (c, s, e) in enumerate(vocab.itertuples(index=False))}
    p169k_key_to_id = {(c, int(s), int(e)): int(i) for (c, s, e), i in zip(p169k.itertuples(index=False), p169k_ids)}

    intervals_all = load_gene_intervals(os.path.join(a.reference, "gene_intervals.csv"))
    gex_var = ad.read_h5ad(os.path.join(raw10k, "pbmc10k_coupled_gex.h5ad"), backed="r").var
    gene_ids, gene_symbols = hvg_genes(gex_var, os.path.join(a.reference, "hvg_genes.csv"), set(intervals_all["gene_id"]))
    intervals = select_gene_intervals(intervals_all, gene_ids)
    log(f"[genes] {len(gene_ids)} genes")

    build_pbmc10k(atac, os.path.join(raw10k, "pbmc10k_coupled_gex.h5ad"), gene_ids, gene_symbols, intervals, vocab,
                  key_to_id, out10k)
    build_pbmc169k(batches, vocab, p169k_key_to_id, out169k)
    if not a.skip_embeddings:
        from .alphagenome import gene_embeddings, peak_embeddings

        peak_embeddings(vocab, fasta, a.weights, os.path.join(shared, "peak_embeddings_alphagenome.h5"))
        gene_embeddings(intervals, gene_ids, gene_symbols, fasta, a.weights,
                        os.path.join(out10k, "gene_embeddings_alphagenome.h5"))
    log(f"[done] {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
