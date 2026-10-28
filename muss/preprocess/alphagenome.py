"""AlphaGenome sequence embeddings of peaks and genes (the fixed token inputs of MUSS).

Requires the PyTorch port of AlphaGenome (https://github.com/genomicsxai/alphagenome-pytorch,
commit 6e96bb5) and its all-folds weights ``model_all_folds.safetensors``
(https://huggingface.co/gtca/alphagenome_pytorch).

The embedding of an interval is the mean of the 128-bp token embeddings (``embeddings_128bp``,
3,072-d) that overlap it. The model sees real genomic context only (no N padding): every input
window lies inside the chromosome and is 2,048-131,072 bp long, a multiple of 2,048 bp.

* peaks: peaks longer than 131,072 bp are cut into pieces; the pieces of a chromosome are then
  greedily grouped (in coordinate order) into the fewest windows, each as short as possible.
* genes: every <=131,072-bp piece of the gene interval gets its own window, centered on it.
Pieces of one interval are combined by a token-count weighted mean.

Output HDF5: ``peak_embeddings`` / ``gene_embeddings`` [N, 3072] float32, ``n_128bp_tokens`` [N],
plus the interval coordinates.
"""

import time
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from .common import log

TOKEN_BP = 128
MAX_WINDOW_BP = 131_072
MIN_WINDOW_BP = 2_048
WINDOW_MULTIPLE_BP = 2_048
HUMAN = 0  # AlphaGenome organism index


class Fasta:
    def __init__(self, path: str):
        from pyfaidx import Fasta as _Fasta

        self._fa = _Fasta(path, as_raw=True, sequence_always_upper=True)

    def __call__(self, chrom: str, start: int, end: int) -> str:
        return str(self._fa[str(chrom)][max(0, int(start)):int(end)]).upper()

    def chrom_size(self, chrom: str) -> int:
        return int(len(self._fa[str(chrom)]))


def _round_up(x: int, multiple: int) -> int:
    return ((int(x) + multiple - 1) // multiple) * multiple


def window_len(span_bp: int) -> int:
    return max(MIN_WINDOW_BP, _round_up(span_bp, WINDOW_MULTIPLE_BP))


def context_window(chrom_len: int, start: int, end: int, input_len: int) -> Tuple[int, int]:
    """Window of ``input_len`` bp containing [start, end), centered on it and kept inside the chromosome."""
    s, e = (int(end), int(start)) if int(end) < int(start) else (int(start), int(end))
    e = max(e, s + 1)
    if input_len < e - s:
        raise ValueError(f"window {input_len} bp shorter than interval ({s}, {e})")
    if chrom_len < input_len:
        raise RuntimeError(f"chromosome ({chrom_len} bp) shorter than the window ({input_len} bp)")
    ws = max(0, min(s - (input_len - (e - s)) // 2, chrom_len - input_len))
    return int(ws), int(ws + input_len)


def token_range(window_start: int, start: int, end: int, n_tokens: int) -> Tuple[int, int]:
    """[i0, i1): tokens (128-bp bins of the window) overlapping [start, end)."""
    i0 = (int(start) - int(window_start)) // TOKEN_BP
    i1 = (int(end) - int(window_start) + TOKEN_BP - 1) // TOKEN_BP
    i0 = max(0, min(i0, n_tokens - 1))
    return int(i0), int(max(i0 + 1, min(i1, n_tokens)))


def _pieces(start: int, end: int) -> List[Tuple[int, int]]:
    s, e = (int(end), int(start)) if int(end) < int(start) else (int(start), int(end))
    e = max(e, s + 1)
    return [(c, min(e, c + MAX_WINDOW_BP)) for c in range(s, e, MAX_WINDOW_BP)]


def plan_peak_windows(peaks: pd.DataFrame, chrom_size) -> Dict[int, List[Tuple[str, int, int, List[Tuple[int, int, int]]]]]:
    """input_len -> [(chrom, window_start, window_end, [(peak_idx, piece_start, piece_end), ...]), ...]."""
    jobs: Dict[int, list] = {}
    peaks = peaks[["Chromosome", "Start", "End"]].reset_index(drop=True)
    for chrom, dfc in peaks.groupby("Chromosome", sort=False):
        chrom, chrom_len = str(chrom), chrom_size(str(chrom))
        pieces = sorted(
            (cs, ce, int(i)) for i in dfc.index.tolist()
            for cs, ce in _pieces(peaks.at[i, "Start"], peaks.at[i, "End"])
        )
        i = 0
        while i < len(pieces):
            g_start, g_end = pieces[i][0], pieces[i][1]
            j = i + 1
            while j < len(pieces):
                cand_end = max(g_end, pieces[j][1])
                if window_len(cand_end - g_start) > MAX_WINDOW_BP:
                    break
                g_end, j = cand_end, j + 1
            L = window_len(g_end - g_start)
            ws, we = context_window(chrom_len, g_start, g_end, L)
            jobs.setdefault(L, []).append((chrom, ws, we, [(idx, cs, ce) for cs, ce, idx in pieces[i:j]]))
            i = j
    return jobs


def plan_interval_windows(intervals: pd.DataFrame, chrom_size) -> Dict[int, List[Tuple[int, str, int, int, int, int]]]:
    """input_len -> [(item_idx, chrom, window_start, window_end, piece_start, piece_end), ...]."""
    jobs: Dict[int, list] = {}
    for idx in range(len(intervals)):
        chrom = str(intervals["Chromosome"].iat[idx])
        chrom_len = chrom_size(chrom)
        for cs, ce in _pieces(intervals["Start"].iat[idx], intervals["End"].iat[idx]):
            L = window_len(max(1, ce - cs))
            ws, we = context_window(chrom_len, cs, ce, L)
            jobs.setdefault(L, []).append((idx, chrom, ws, we, cs, ce))
    return jobs


class _Encoder:
    def __init__(self, weights: str):
        import torch
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

        self.torch, self.onehot = torch, sequence_to_onehot_tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"[alphagenome] loading {weights} on {self.device}")
        self.model = AlphaGenome.from_pretrained(str(weights)).eval().to(self.device)

    def __call__(self, seqs: Sequence[str], input_len: int):
        """[B, input_len / 128, 3072] float32 token embeddings (on the model device)."""
        torch = self.torch
        for s in seqs:
            if len(s) != input_len:
                raise RuntimeError(f"sequence of {len(s)} bp, expected {input_len}")
        x = torch.stack([self.onehot(s).to(torch.float32) for s in seqs], dim=0).contiguous()
        x = x.pin_memory().to(self.device, non_blocking=True) if self.device.type == "cuda" else x.to(self.device)
        org = torch.full((len(seqs),), HUMAN, dtype=torch.long, device=self.device)
        tokens = self.model.encode(x, organism_index=org, resolutions=(TOKEN_BP,))[f"embeddings_{TOKEN_BP}bp"]
        tokens = tokens.to(torch.float32)
        if tokens.shape[1] != input_len // TOKEN_BP:
            raise RuntimeError(f"unexpected token count {tokens.shape[1]} for a {input_len}-bp window")
        return tokens


def _new_h5(path: str, dset: str, n: int, dim: int, coords: pd.DataFrame, pooling: str, windowing: str):
    h5 = h5py.File(path, "w")
    emb = h5.create_dataset(dset, shape=(n, dim), dtype="f4", chunks=(min(1024, max(1, n)), dim),
                            fillvalue=np.float32(0.0))
    cnt = h5.create_dataset("n_128bp_tokens", shape=(n,), dtype="i8", chunks=(min(8192, max(1, n)),),
                            fillvalue=np.int64(0))
    h5.create_dataset("Chromosome", data=np.array(coords["Chromosome"].astype(str).tolist(), dtype="S"))
    h5.create_dataset("Start", data=coords["Start"].astype(np.int64).to_numpy())
    h5.create_dataset("End", data=coords["End"].astype(np.int64).to_numpy())
    for k, v in {"hidden_dim": dim, "feature_space": f"embeddings_{TOKEN_BP}bp", "pooled": pooling,
                 "token_bp": TOKEN_BP, "max_chunk_bp": MAX_WINDOW_BP, "padding": "none", "windowing": windowing,
                 "min_input_bp": MIN_WINDOW_BP, "input_multiple_bp": WINDOW_MULTIPLE_BP,
                 "organism_index": HUMAN, "model_name": "model_all_folds.safetensors"}.items():
        h5.attrs[k] = v
    return h5, emb, cnt


def _accumulate(emb, cnt, ids: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> None:
    uniq, inv = np.unique(ids, return_inverse=True)
    s = np.zeros((uniq.size, sums.shape[1]), dtype=np.float32)
    np.add.at(s, inv, sums)
    c = np.zeros((uniq.size,), dtype=np.int64)
    np.add.at(c, inv, counts)
    prev = emb[uniq, :]
    prev += s
    emb[uniq, :] = prev
    cnt[uniq] = cnt[uniq] + c


def _finalize(emb, cnt, n: int, block: int = 2048) -> int:
    """Divide the token sums by the token counts; returns the number of intervals without tokens."""
    n_missing = 0
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)
        arr, c = emb[b0:b1, :], cnt[b0:b1]
        mask = c > 0
        n_missing += int((~mask).sum())
        if mask.any():
            arr[mask, :] = arr[mask, :] / c[mask, None].astype(np.float32)
        emb[b0:b1, :] = arr
    return n_missing


def peak_embeddings(peaks: pd.DataFrame, fasta_path: str, weights: str, out_h5: str, batch_size: int = 8) -> None:
    """Embed every peak (rows of ``peaks``: Chromosome, Start, End) into ``out_h5``/peak_embeddings."""
    import torch

    t0 = time.time()
    peaks = peaks[["Chromosome", "Start", "End"]].copy().reset_index(drop=True)
    n = len(peaks)
    fasta = Fasta(fasta_path)
    jobs = plan_peak_windows(peaks, fasta.chrom_size)
    n_windows = sum(len(v) for v in jobs.values())
    log(f"[alphagenome] {n} peaks in {n_windows} windows")
    encode = _Encoder(weights)
    h5 = emb = cnt = None
    done = 0
    with torch.inference_mode():
        for L in sorted(jobs):
            n_tok = L // TOKEN_BP
            for j0 in range(0, len(jobs[L]), batch_size):
                batch = jobs[L][j0:j0 + batch_size]
                tokens = encode([fasta(c, ws, we) for c, ws, we, _ in batch], L)
                if h5 is None:
                    h5, emb, cnt = _new_h5(out_h5, "peak_embeddings", n, int(tokens.shape[-1]), peaks,
                                           f"mean_over_{TOKEN_BP}bp_tokens_overlapping_original_peak",
                                           "shared_real_genome_context_windows_covering_multiple_peaks")
                ids, sums, counts = [], [], []
                for b, (_, ws, _, pieces) in enumerate(batch):
                    y = tokens[b]
                    csum = torch.cumsum(torch.cat([y.new_zeros((1, y.shape[1])), y], dim=0), dim=0)
                    rng = [token_range(ws, ps, pe, n_tok) for _, ps, pe in pieces]
                    i0 = torch.as_tensor([r[0] for r in rng], device=y.device, dtype=torch.long)
                    i1 = torch.as_tensor([r[1] for r in rng], device=y.device, dtype=torch.long)
                    sums.append((csum.index_select(0, i1) - csum.index_select(0, i0)).cpu().numpy().astype(np.float32, copy=False))
                    counts.append(np.asarray([r[1] - r[0] for r in rng], dtype=np.int64))
                    ids.extend(p for p, _, _ in pieces)
                _accumulate(emb, cnt, np.asarray(ids, dtype=np.int64), np.concatenate(sums, 0), np.concatenate(counts, 0))
                done += len(batch)
            log(f"[alphagenome] windows {done}/{n_windows} ({(time.time() - t0) / 60:.1f} min)")
    n_missing = _finalize(emb, cnt, n)
    h5.attrs["n_peaks"] = n
    h5.close()
    if n_missing:
        raise RuntimeError(f"{n_missing} peaks received no tokens")
    log(f"[alphagenome] saved {out_h5} ({(time.time() - t0) / 60:.1f} min)")


def gene_embeddings(intervals: pd.DataFrame, gene_ids: Sequence[str], gene_symbols: Sequence[str], fasta_path: str,
                    weights: str, out_h5: str, batch_size: int = 2) -> None:
    """Embed every gene interval (rows of ``intervals``: Chromosome, Start, End) into ``out_h5``/gene_embeddings."""
    import torch

    t0 = time.time()
    intervals = intervals[["Chromosome", "Start", "End"]].copy().reset_index(drop=True)
    n = len(intervals)
    fasta = Fasta(fasta_path)
    jobs = plan_interval_windows(intervals, fasta.chrom_size)
    encode = _Encoder(weights)
    h5 = emb = cnt = None
    with torch.inference_mode():
        for L in sorted(jobs):
            n_tok = L // TOKEN_BP
            for j0 in range(0, len(jobs[L]), batch_size):
                batch = jobs[L][j0:j0 + batch_size]
                tokens = encode([fasta(c, ws, we) for _, c, ws, we, _, _ in batch], L)
                if h5 is None:
                    h5, emb, cnt = _new_h5(out_h5, "gene_embeddings", n, int(tokens.shape[-1]), intervals,
                                           f"mean_over_{TOKEN_BP}bp_tokens_overlapping_original_interval",
                                           "real_genome_context_window_contains_interval")
                    h5.create_dataset("gene_ids", data=np.array([str(g).encode() for g in gene_ids], dtype="S"))
                    h5.create_dataset("gene_symbols", data=np.array([str(g).encode() for g in gene_symbols], dtype="S"))
                    h5.attrs["n_genes"] = n
                sums = np.zeros((len(batch), tokens.shape[-1]), dtype=np.float32)
                counts = np.zeros((len(batch),), dtype=np.int64)
                for b, (_, _, ws, _, cs, ce) in enumerate(batch):
                    i0, i1 = token_range(ws, cs, ce, n_tok)
                    sums[b] = tokens[b, i0:i1, :].sum(dim=0).cpu().numpy().astype(np.float32, copy=False)
                    counts[b] = i1 - i0
                _accumulate(emb, cnt, np.asarray([job[0] for job in batch], dtype=np.int64), sums, counts)
    _finalize(emb, cnt, n)
    h5.close()
    log(f"[alphagenome] saved {out_h5} ({(time.time() - t0) / 60:.1f} min)")
