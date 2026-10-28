"""Building blocks shared by the preprocessing pipelines.

A processed GEX dataset has the layout read by :mod:`muss.data`::

    <out>/chunks/overview.pkl               peak / gene vocabularies and split positions
    <out>/chunks/<split>_overview.pkl       chunk index of one split
    <out>/chunks/<split>_<k>.pkl            list of {"idx": cell position, "data": payload}
    <out>/peak_embeddings_alphagenome.h5    AlphaGenome embedding per peak   (dataset "peak_embeddings")
    <out>/gene_embeddings_alphagenome.h5    AlphaGenome embedding per gene   (dataset "gene_embeddings")
    <out>/split_id.json                     split definition (cell ids)

The per-cell payload is ``(gex float32[G], peak_ids int32[P], (celltype, barcode),
expressed_gene_ids int32, expressed_counts int32)``; peak ids index the dataset's peak
vocabulary and are sorted and unique.
"""

import json
import os
import pickle
import re
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CANONICAL_CHROM_KEYS = [str(i) for i in range(1, 23)] + ["X", "Y", "M"]
_CANONICAL = set(CANONICAL_CHROM_KEYS)
PAYLOAD_FORMAT = "tuple(GEX, peak_ids, meta_tuple, expr_gene_ids, expr_vals)"
META_FORMAT = "tuple(celltype, barcode)"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def strip_version(gene_id: Any) -> str:
    """ENSG00000187634.12 -> ENSG00000187634."""
    return str(gene_id).split(".", 1)[0]


# ----------------------------------------------------------------------------
# chromosomes
# ----------------------------------------------------------------------------
def chrom_key(chrom: Any) -> str:
    """'chr1' / '1' -> '1', 'chrx' -> 'X', 'chrMT' -> 'M'."""
    s = re.sub(r"^chr", "", str(chrom).strip(), flags=re.IGNORECASE).upper()
    return "M" if s == "MT" else s


def canonical_chrom(chrom: Any) -> Optional[str]:
    """'chr<key>' for chr1-22, X, Y, M; None for any other contig (scaffolds, decoys, ...)."""
    k = chrom_key(chrom)
    return f"chr{k}" if k in _CANONICAL else None


def with_chr_prefix(chrom: Any) -> str:
    s = str(chrom).strip()
    return s if s.lower().startswith("chr") else f"chr{s}"


def check_fasta_has_canonical_chroms(fasta_path: str) -> None:
    """Every canonical chromosome must be in the FASTA (its sequence is embedded)."""
    fai = fasta_path + ".fai"
    if not os.path.exists(fai):
        raise FileNotFoundError(f"FASTA index not found: {fai} (run `samtools faidx {fasta_path}`)")
    with open(fai) as f:
        names = {line.split("\t", 1)[0] for line in f if line.strip()}
    missing = [f"chr{k}" for k in CANONICAL_CHROM_KEYS if f"chr{k}" not in names]
    if missing:
        raise RuntimeError(f"{fasta_path} lacks canonical chromosomes {missing}")


def chrom_id_map(chroms: Iterable[Any]) -> Dict[str, int]:
    """chr1..22, X, Y, M -> 1..25; other contigs get the next ids in order of appearance."""
    base = {c: i + 1 for i, c in enumerate(CANONICAL_CHROM_KEYS)}
    out: Dict[str, int] = {}
    next_id = len(CANONICAL_CHROM_KEYS) + 1
    for c in chroms:
        k = chrom_key(c)
        if k in out:
            continue
        if k in base:
            out[k] = base[k]
        else:
            out[k] = next_id
            next_id += 1
    return out


def chrom_ids(chroms: Sequence[Any], id_map: Dict[str, int]) -> List[int]:
    return [int(id_map.get(chrom_key(c), 0)) for c in chroms]


def peaks_meta(peaks: pd.DataFrame, id_map: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Peak vocabulary as stored in overview.pkl."""
    chrom = peaks["Chromosome"].astype(str).tolist()
    id_map = id_map if id_map is not None else chrom_id_map(chrom)
    return {
        "Chromosome": chrom,
        "Chromosome_id": chrom_ids(chrom, id_map),
        "Start": peaks["Start"].astype(int).tolist(),
        "End": peaks["End"].astype(int).tolist(),
        "n_peaks": int(len(peaks)),
    }


# ----------------------------------------------------------------------------
# genes
# ----------------------------------------------------------------------------
def load_gene_intervals(path: str) -> pd.DataFrame:
    """Gene interval table (one transcript per gene): Chromosome, Start, End, Strand, gene_id[, gene_type]."""
    df = pd.read_csv(path, sep="\t" if path.endswith((".tsv", ".bed")) else ",")
    for c in ("Chromosome", "Start", "End", "Strand", "gene_id"):
        if c not in df.columns:
            raise KeyError(f"{path} lacks column {c}")
    df["Chromosome"] = [canonical_chrom(c) or with_chr_prefix(c) for c in df["Chromosome"]]
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce").astype("Int64").astype("int64")
    df["End"] = pd.to_numeric(df["End"], errors="coerce").astype("Int64").astype("int64")
    df["Strand"] = df["Strand"].astype(str).str.strip()
    df["gene_id"] = df["gene_id"].map(strip_version)
    return df


def select_gene_intervals(intervals: pd.DataFrame, gene_ids: Sequence[str]) -> pd.DataFrame:
    """One interval per gene (protein-coding first, then leftmost), in the order of ``gene_ids``."""
    cols = ["Chromosome", "Start", "End", "Strand", "gene_id"]
    df = intervals[cols + (["gene_type"] if "gene_type" in intervals.columns else [])].copy()
    df = df[df["gene_id"].isin(set(map(str, gene_ids)))].copy()
    if "gene_type" in df.columns:
        df["_rank"] = (df["gene_type"] != "protein_coding").astype(int)
        df = df.sort_values(["gene_id", "_rank", "Start"]).drop_duplicates(subset=["gene_id"], keep="first")
    else:
        df = df.sort_values(["gene_id", "Start"]).drop_duplicates(subset=["gene_id"], keep="first")
    order = {g: i for i, g in enumerate(gene_ids)}
    df["_ord"] = df["gene_id"].map(order)
    df = df.sort_values("_ord").reset_index(drop=True)[cols]
    if len(df) != len(gene_ids):
        present = set(df["gene_id"])
        missing = [g for g in gene_ids if g not in present]
        raise RuntimeError(f"{len(missing)} genes have no interval, e.g. {missing[:5]}")
    return df


def genes_meta(gene_ids: Sequence[str], gene_symbols: Sequence[str], intervals: pd.DataFrame,
               id_map: Dict[str, int]) -> Dict[str, Any]:
    """Gene vocabulary as stored in overview.pkl (TSS = interval start on +, end on - strand)."""
    tss = np.where(intervals["Strand"] == "+", intervals["Start"].to_numpy(), intervals["End"].to_numpy())
    chrom = intervals["Chromosome"].astype(str).tolist()
    return {
        "gene_ids": list(gene_ids),
        "gene_symbols": list(gene_symbols),
        "n_genes": int(len(gene_ids)),
        "tss": tss.astype(np.int64).tolist(),
        "Chromosome": chrom,
        "Chromosome_id": chrom_ids(chrom, id_map),
    }


def load_reference_genes(reference_dir: str) -> Dict[str, Any]:
    """Gene vocabulary of the processed PBMC10K reference dataset."""
    genes = load_pickle(os.path.join(reference_dir, "chunks", "overview.pkl"))["genes"]
    ids = [strip_version(g) for g in genes["gene_ids"]]
    if len(set(ids)) != len(ids):
        raise ValueError("reference gene vocabulary has duplicate gene ids")
    return {k: list(genes[k]) for k in ("gene_ids", "gene_symbols", "tss", "Chromosome", "Chromosome_id")} | {
        "gene_ids": ids, "n_genes": len(ids)}


def copy_reference_gene_embeddings(reference_dir: str, out_dir: str, gene_ids: Sequence[str]) -> None:
    """The fine-tuning datasets share the reference gene vocabulary, hence its gene embeddings."""
    import h5py

    src = os.path.join(reference_dir, "gene_embeddings_alphagenome.h5")
    with h5py.File(src, "r") as h5:
        src_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["gene_ids"][:]]
    if src_ids != list(gene_ids):
        raise ValueError(f"gene ids of {src} differ from the dataset's gene vocabulary")
    dst = os.path.join(out_dir, "gene_embeddings_alphagenome.h5")
    shutil.copyfile(src, dst)
    log(f"[genes] copied reference gene embeddings -> {dst}")


def dense_gene_index(source_gene_ids: Sequence[str], gene_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(positions in the vocabulary, columns in the source matrix) of the vocabulary genes the source measures.

    A source gene id listed twice maps to its first column; vocabulary genes missing from
    the source stay zero.
    """
    first: Dict[str, int] = {}
    for i, g in enumerate(source_gene_ids):
        first.setdefault(str(g), i)
    src = np.array([first.get(str(g), -1) for g in gene_ids], dtype=np.int64)
    present = src >= 0
    return np.flatnonzero(present).astype(np.int64), src[present]


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------
def _split_sizes(n: int, valid_ratio: float, test_ratio: float) -> Tuple[int, int]:
    n_valid, n_test = int(round(valid_ratio * n)), int(round(test_ratio * n))
    if n_valid + n_test > n:
        overflow = n_valid + n_test - n
        dec = min(overflow, n_test)
        n_test -= dec
        n_valid = max(0, n_valid - (overflow - dec))
    return n_valid, n_test


def split_unseen_forced(cell_ids: Sequence[str], celltypes: Sequence[str], valid_ratio: float,
                        test_ratio: float, unseen_celltypes: Sequence[str], seed: int = 42) -> Dict[str, List[str]]:
    """Random train/valid/test split; cells of ``unseen_celltypes`` go to valid/test only."""
    n = len(cell_ids)
    unseen = set(unseen_celltypes)
    rng = np.random.RandomState(int(seed))
    idx = np.arange(n, dtype=np.int64)
    is_unseen = np.fromiter((ct in unseen for ct in celltypes), dtype=bool, count=n)
    unseen_idx, seen_idx = idx[is_unseen], idx[~is_unseen]
    rng.shuffle(unseen_idx)
    rng.shuffle(seen_idx)
    n_valid_unseen = int(round(valid_ratio / (valid_ratio + test_ratio) * len(unseen_idx)))
    n_valid, n_test = _split_sizes(len(seen_idx), valid_ratio, test_ratio)
    valid = np.sort(np.concatenate([unseen_idx[:n_valid_unseen], seen_idx[:n_valid]]))
    test = np.sort(np.concatenate([unseen_idx[n_valid_unseen:], seen_idx[n_valid:n_valid + n_test]]))
    train = np.sort(seen_idx[n_valid + n_test:])
    ids = [str(c) for c in cell_ids]
    return {name: [ids[i] for i in pos] for name, pos in (("train_idx", train), ("valid_idx", valid), ("test_idx", test))}


def split_stratified(cell_ids: Sequence[str], celltypes: Sequence[str], valid_ratio: float, test_ratio: float,
                     unseen_celltypes: Sequence[str], seed: int = 42) -> Dict[str, List[str]]:
    """Like :func:`split_unseen_forced`, but the seen cell types are split one by one (sorted by name)."""
    unseen = set(unseen_celltypes)
    rng = np.random.RandomState(int(seed))
    by_ct: Dict[str, List[int]] = {}
    for i, ct in enumerate(celltypes):
        by_ct.setdefault(str(ct), []).append(i)
    train, valid, test = [], [], []
    for ct in sorted(by_ct):
        pos = list(by_ct[ct])
        rng.shuffle(pos)
        if ct in unseen:
            k = int(round(valid_ratio / (valid_ratio + test_ratio) * len(pos)))
            valid += pos[:k]
            test += pos[k:]
        else:
            n_valid, n_test = _split_sizes(len(pos), valid_ratio, test_ratio)
            valid += pos[:n_valid]
            test += pos[n_valid:n_valid + n_test]
            train += pos[n_valid + n_test:]
    ids = [str(c) for c in cell_ids]
    return {name: [ids[i] for i in sorted(pos)] for name, pos in (("train_idx", train), ("valid_idx", valid), ("test_idx", test))}


def split_random(n: int, valid_ratio: float, test_ratio: float, seed: int = 42) -> Dict[str, List[int]]:
    """Random positional split of n cells (PBMC169K)."""
    n_valid, n_test = _split_sizes(n, valid_ratio, test_ratio)
    idx = np.arange(n, dtype=np.int64)
    np.random.RandomState(int(seed)).shuffle(idx)
    return {"train_idx": sorted(idx[n_valid + n_test:].tolist()), "valid_idx": sorted(idx[:n_valid].tolist()),
            "test_idx": sorted(idx[n_valid:n_valid + n_test].tolist())}


def split_positions(splits: Dict[str, List[str]], cell_ids: Sequence[str]) -> Dict[str, List[int]]:
    """Cell-id split -> positions in ``cell_ids``; every listed cell must exist and be in one split only."""
    pos_of: Dict[str, int] = {}
    for i, c in enumerate(cell_ids):
        pos_of.setdefault(str(c), i)
    out: Dict[str, List[int]] = {}
    seen: Dict[int, str] = {}
    for name, ids in splits.items():
        out[name] = []
        for c in ids:
            if c not in pos_of:
                raise KeyError(f"split '{name}' lists cell {c!r}, which is not in the dataset")
            p = pos_of[c]
            if seen.setdefault(p, name) != name:
                raise ValueError(f"cell {c!r} is in more than one split")
            out[name].append(p)
    return out


# ----------------------------------------------------------------------------
# chunk writing
# ----------------------------------------------------------------------------
def expr_ranked_lists(gex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Expressed genes (ids, rounded counts), by decreasing count.

    MUSS does not read these two payload fields; they are kept so that the chunk files have
    the published format.
    """
    gx = np.asarray(gex).ravel()
    gx_int = np.rint(gx).astype(np.int32)
    ids = np.flatnonzero(gx_int > 0).astype(np.int32)
    vals = gx_int[ids]
    if ids.size > 1:
        order = np.argsort(-vals, kind="mergesort")
        ids, vals = ids[order], vals[order]
    return ids, vals


class ChunkWriter:
    """Streams the cells of one split into pickles of at most ``max_chunk_size_mb`` each."""

    def __init__(self, split: str, chunks_dir: str, max_chunk_size_mb: int = 1024):
        self.split, self.chunks_dir = split, chunks_dir
        self.max_bytes = int(max_chunk_size_mb) * 1024 * 1024
        self.overview: Dict[str, Any] = {"split": split, "max_chunk_size_mb": int(max_chunk_size_mb), "chunks": []}
        self._reset()

    def _reset(self):
        self.samples, self.indices, self.n_peaks, self.size = [], [], [], 0

    def _flush(self):
        if not self.samples:
            return
        k = len(self.overview["chunks"])
        fname = f"{self.split}_{k}.pkl"
        write_pickle(self.samples, os.path.join(self.chunks_dir, fname))
        self.overview["chunks"].append({"chunk_id": k, "file": fname, "n_samples": len(self.samples),
                                        "sample_indices": self.indices, "n_peaks_per_sample": self.n_peaks,
                                        "size_bytes": self.size})
        self._reset()

    def add(self, idx: int, payload: Any, n_peaks: int) -> None:
        sample = {"idx": int(idx), "data": payload}
        nbytes = len(pickle.dumps(sample, protocol=4))
        if self.samples and self.size + nbytes > self.max_bytes:
            self._flush()
        self.samples.append(sample)
        self.indices.append(int(idx))
        self.n_peaks.append(int(n_peaks))
        self.size += nbytes

    def finalize(self) -> Dict[str, Any]:
        self._flush()
        return self.overview


class GEXDatasetWriter:
    """Writes the chunk files of a GEX dataset; ``add`` must be called for every split position."""

    def __init__(self, out_dir: str, positions: Dict[str, List[int]], max_chunk_size_mb: int = 1024):
        self.out_dir = out_dir
        self.chunks_dir = os.path.join(out_dir, "chunks")
        os.makedirs(self.chunks_dir, exist_ok=True)
        self.positions = {(k[:-4] if k.endswith("_idx") else k): v for k, v in positions.items()}
        self.split_of = {int(p): name for name, pos in self.positions.items() for p in pos}
        self.writers = {name: ChunkWriter(name, self.chunks_dir, max_chunk_size_mb)
                        for name, pos in self.positions.items() if pos}
        self.n_written = 0

    def add(self, idx: int, gex: np.ndarray, peak_ids: np.ndarray, celltype: str, barcode: str) -> None:
        gex = np.asarray(gex, dtype=np.float32)
        peak_ids = np.asarray(peak_ids, dtype=np.int32)
        expr_ids, expr_vals = expr_ranked_lists(gex)
        payload = (gex, peak_ids, (str(celltype), str(barcode)), expr_ids, expr_vals)
        self.writers[self.split_of[int(idx)]].add(idx, payload, n_peaks=int(peak_ids.size))
        self.n_written += 1

    def finalize(self, genes: Dict[str, Any], peaks: Dict[str, Any], dataset_meta: Dict[str, Any]) -> Dict[str, Any]:
        n_expected = sum(len(v) for v in self.positions.values())
        if self.n_written != n_expected:
            raise RuntimeError(f"wrote {self.n_written} cells, splits list {n_expected}")
        split_overviews = {}
        for name, w in self.writers.items():
            split_overviews[name] = w.finalize()
            write_pickle(split_overviews[name], os.path.join(self.chunks_dir, f"{name}_overview.pkl"))
        overview = {
            "format": "muss-processed-v1",
            "n_cells": int(self.n_written),
            "n_genes": int(genes["n_genes"]),
            "n_peaks": int(peaks["n_peaks"]),
            "genes": genes,
            "peaks": peaks,
            "splits": self.positions,
            "split_overviews": split_overviews,
            "dataset_meta": dataset_meta,
            "cell_payload_format": PAYLOAD_FORMAT,
            "cell_meta_format": META_FORMAT,
        }
        write_pickle(overview, os.path.join(self.chunks_dir, "overview.pkl"))
        log(f"[write] {self.n_written} cells -> {self.chunks_dir} "
            + ", ".join(f"{k}={len(v)}" for k, v in self.positions.items()))
        return overview
