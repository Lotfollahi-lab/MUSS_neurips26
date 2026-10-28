"""Dataset metadata, embeddings and deterministic cell ordering."""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

N_SPECIAL = 5  # [pad, unk, bos, eos, mask]


class PeakDictionary:
    """Token vocabulary: 5 special tokens [pad, unk, bos, eos, mask] (ids 0-4), then one token
    per peak (token = peak_id + 5). Reserved IDs are retained for checkpoint compatibility."""

    def __init__(self, n_peaks: int):
        self.n_peaks = int(n_peaks)
        self.n_special = N_SPECIAL

    def pad(self) -> int:
        return 0

    def __len__(self) -> int:
        return self.n_special + self.n_peaks


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_embeddings(h5_path: str, dset: str) -> np.ndarray:
    """Load a full [N, D] embedding matrix from an HDF5 file as float16."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"embedding file not found: {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        if dset not in h5:
            raise KeyError(f'{h5_path} has no dataset "{dset}"')
        arr = h5[dset][:]
    logger.info("loaded %s %s from %s", dset, arr.shape, h5_path)
    return np.asarray(arr, dtype=np.float16)


# ----------------------------------------------------------------------------
# overview metadata
# ----------------------------------------------------------------------------
def peak_meta(overview: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Peak coordinates of the vocabulary stored in a dataset overview."""
    peaks = overview["peaks"]
    starts = np.asarray(peaks["Start"], dtype=np.int64)
    ends = np.asarray(peaks["End"], dtype=np.int64)
    chrom = np.asarray(peaks["Chromosome"], dtype=object)
    n = int(overview.get("n_peaks", peaks.get("n_peaks", len(starts))))
    chrom_id = np.asarray(peaks["Chromosome_id"], dtype=np.int32)
    if not (starts.shape[0] == ends.shape[0] == chrom.shape[0] == chrom_id.shape[0] == n):
        raise ValueError("overview peak metadata length mismatch")
    return {"n_peaks": n, "Chromosome": chrom, "Chromosome_id": chrom_id, "Start": starts, "End": ends}


def gene_meta(overview: Dict[str, Any]) -> Dict[str, Any]:
    """Gene ids, TSS and chromosome ids of the gene panel stored in a dataset overview."""
    genes = overview["genes"]
    n = int(overview.get("n_genes", genes.get("n_genes", len(genes.get("gene_ids", [])))))
    tss = np.asarray(genes["tss"], dtype=np.int64)
    chrom_id = np.asarray(genes["Chromosome_id"], dtype=np.int32)
    if tss.shape[0] != n or chrom_id.shape[0] != n:
        raise ValueError("overview gene metadata length mismatch")
    return {
        "n_genes": n,
        "gene_ids": list(genes.get("gene_ids", [])),
        "Chromosome": list(genes.get("Chromosome", [])),
        "tss": tss,
        "Chromosome_id": chrom_id,
    }


# ----------------------------------------------------------------------------
# ordering: shuffle groups (chunk files), then sort each group by length
# ----------------------------------------------------------------------------
def grouped_ordered_indices(seed: int, indices_by_group: List[np.ndarray], lengths: np.ndarray) -> np.ndarray:
    """Visit chunk files in a random order; within a file, cells are shuffled and then
    stably sorted by (capped) peak count.

    The order is fixed across epochs and resumed runs. Seed + 1 reproduces the
    ordering used to train the released models.
    """
    rng = np.random.RandomState(seed + 1)
    group_order = np.arange(len(indices_by_group), dtype=np.int64)
    rng.shuffle(group_order)
    out: List[int] = []
    for gid in group_order:
        idxs = indices_by_group[gid]
        if idxs.size == 0:
            continue
        idxs_shuf = idxs[rng.permutation(idxs.size)]
        order = np.argsort(lengths[idxs_shuf].astype(np.int64, copy=False), kind="mergesort")
        out.extend(idxs_shuf[order].tolist())
    return np.asarray(out, dtype=np.int64)


class SingleFileCache:
    """Keeps the most recently used chunk pickle in memory (cells are visited file by file)."""

    def __init__(self):
        self.path: Optional[str] = None
        self.data: Optional[list] = None

    def get(self, path: str) -> list:
        if self.path != path or self.data is None:
            logger.info("loading %s", path)
            self.data = load_pickle(path)
            self.path = path
        return self.data
