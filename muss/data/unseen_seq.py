"""Held-out ("unseen") genomic sequence folds.

``sequences_human.bed`` (Borzoi's human sequence folds) assigns 196,608-bp genomic windows
to the folds ``fold0`` - ``fold7``. Windows of the held-out folds (default ``fold3,fold4``)
are merged per chromosome; a peak is *unseen* if it overlaps a held-out window, and a gene is *unseen*
if its TSS lies inside one. Unseen peaks are removed from training cells; unseen genes are
excluded from the model and the loss during fine-tuning and form the "unseen sequence"
evaluation setting.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Windows = Dict[str, Tuple[np.ndarray, np.ndarray]]


def norm_chr_ucsc(chrom: str) -> str:
    """'1' -> 'chr1', 'MT' -> 'chrM', 'chrx' -> 'chrX'."""
    s = str(chrom).strip()
    if not s:
        return s
    core = (s[3:] if s.lower().startswith("chr") else s).strip()
    up = core.upper()
    if up == "MT":
        up = "M"
    return "chr" + up if up in ("X", "Y", "M") else "chr" + core


def load_holdout_windows(sequences_bed: str, folds: Set[str]) -> Windows:
    if not os.path.exists(sequences_bed):
        raise FileNotFoundError(f"sequences_bed not found: {sequences_bed}")
    if not folds:
        raise ValueError("no held-out folds given")
    starts: Dict[str, List[int]] = defaultdict(list)
    ends: Dict[str, List[int]] = defaultdict(list)
    found = set()
    with open(sequences_bed) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                s, e = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            found.add(parts[3].strip())
            if parts[3].strip() in folds and e > s:
                c = norm_chr_ucsc(parts[0])
                starts[c].append(s)
                ends[c].append(e)
    if folds - found:
        raise ValueError(f"held-out folds {sorted(folds - found)} do not occur in {sequences_bed}")

    merged: Windows = {}
    for c in starts:
        s = np.asarray(starts[c], dtype=np.int64)
        e = np.asarray(ends[c], dtype=np.int64)
        order = np.argsort(s, kind="mergesort")
        s, e = s[order], e[order]
        ms, me = [], []
        cur_s, cur_e = int(s[0]), int(e[0])
        for si, ei in zip(s[1:], e[1:]):
            si, ei = int(si), int(ei)
            if si <= cur_e:
                cur_e = max(cur_e, ei)
            else:
                ms.append(cur_s)
                me.append(cur_e)
                cur_s, cur_e = si, ei
        ms.append(cur_s)
        me.append(cur_e)
        merged[c] = (np.asarray(ms, dtype=np.int64), np.asarray(me, dtype=np.int64))
    logger.info(
        "held-out windows: raw=%d merged=%d chroms=%d folds=%s",
        sum(len(v) for v in starts.values()), sum(v[0].size for v in merged.values()), len(merged), sorted(folds),
    )
    return merged


def overlap_mask(chroms: Sequence[str], start: np.ndarray, end: np.ndarray, windows: Windows) -> np.ndarray:
    """True where the half-open interval [start, end) overlaps a window on the same chromosome."""
    hit = np.zeros(len(start), dtype=bool)
    by_chr: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by_chr[c].append(i)
    for c, idxs in by_chr.items():
        if c not in windows:
            continue
        w_s, w_e = windows[c]
        idx = np.asarray(idxs, dtype=np.int64)
        order = np.argsort(start[idx], kind="mergesort")
        idx = idx[order]
        p_s, p_e = start[idx], end[idx]
        j, m = 0, int(w_s.size)
        for k in range(p_s.size):
            ps, pe = int(p_s[k]), int(p_e[k])
            if pe <= ps:
                continue
            while j < m and int(w_e[j]) <= ps:
                j += 1
            if j >= m:
                break
            if pe > int(w_s[j]) and ps < int(w_e[j]):
                hit[int(idx[k])] = True
    return hit


def unseen_masks(
    sequences_bed: str,
    folds: Set[str],
    peak_chrom: Sequence[str],
    peak_start: np.ndarray,
    peak_end: np.ndarray,
    gene_chrom: Sequence[str],
    gene_tss: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unseen_peak_mask [n_peaks], unseen_gene_mask [n_genes])."""
    windows = load_holdout_windows(sequences_bed, folds)
    peak_chr = np.array([norm_chr_ucsc(x) for x in peak_chrom], dtype=object)
    gene_chr = np.array([norm_chr_ucsc(x) for x in gene_chrom], dtype=object)
    peak_mask = overlap_mask(peak_chr, np.asarray(peak_start, np.int64), np.asarray(peak_end, np.int64), windows)
    tss = np.asarray(gene_tss, dtype=np.int64)
    gene_mask = overlap_mask(gene_chr, tss, tss + 1, windows)
    return peak_mask, gene_mask
