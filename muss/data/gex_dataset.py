"""Paired scATAC + scRNA dataset for gene-expression prediction.

Processed layout (written by ``muss.preprocess``)::

    <data>/chunks/overview.pkl            peak vocabulary + gene panel metadata
    <data>/chunks/{split}_overview.pkl    list of chunk files of a split
    <data>/chunks/{split}_{k}.pkl         list of cells: {"idx": int, "data": payload}

payload = (gex float32[G], peak_ids int32[T] (sorted), (celltype, barcode), expr_gene_ids, expr_vals)

Each item is one cell: the tokens of its accessible peaks (peak_id + 5), their genomic
coordinates, and the expression vector of the gene panel as the target.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import SingleFileCache, gene_meta, grouped_ordered_indices, load_pickle, peak_meta

logger = logging.getLogger(__name__)


class MUSSGEXDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        *,
        pad_idx: int,
        n_special: int,
        max_cell_peaks_num: int,
        seed: int,
        unseen_peak_mask: Optional[np.ndarray] = None,
        unseen_gene_mask: Optional[np.ndarray] = None,
        chrom_pad_val: int = 29,
        coord_pad_val: int = -1,
    ):
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.split = split
        self.pad_idx = int(pad_idx)
        self.n_special = int(n_special)
        self.max_cell_peaks_num = int(max_cell_peaks_num)
        self.seed = int(seed)
        self.chrom_pad_val = int(chrom_pad_val)
        self.coord_pad_val = int(coord_pad_val)

        self.chunks_dir = os.path.join(self.data_dir, "chunks")
        overview = load_pickle(os.path.join(self.chunks_dir, "overview.pkl"))
        pm, gm = peak_meta(overview), gene_meta(overview)
        self._n_peaks, self._num_genes = pm["n_peaks"], gm["n_genes"]
        self._peak_starts, self._peak_ends, self._peak_chr_id = pm["Start"], pm["End"], pm["Chromosome_id"]
        self._gene_tss_tensor = torch.as_tensor(gm["tss"], dtype=torch.long)
        self._gene_chr_id_tensor = torch.as_tensor(gm["Chromosome_id"], dtype=torch.long)

        # unseen-sequence peaks are removed from training cells only
        self._unseen_peak_mask = None
        if unseen_peak_mask is not None:
            unseen_peak_mask = np.asarray(unseen_peak_mask, dtype=bool)
            if unseen_peak_mask.shape != (self._n_peaks,):
                raise ValueError("unseen_peak_mask does not match the peak vocabulary")
            self._unseen_peak_mask = unseen_peak_mask
        self._unseen_gene_mask_tensor = None
        if unseen_gene_mask is not None:
            unseen_gene_mask = np.asarray(unseen_gene_mask, dtype=bool)
            if unseen_gene_mask.shape != (self._num_genes,):
                raise ValueError("unseen_gene_mask does not match the gene panel")
            self._unseen_gene_mask_tensor = torch.from_numpy(unseen_gene_mask)

        # ---- sample index over the chunk files of this split ----
        self._chunk_infos: List[Dict[str, Any]] = list(
            load_pickle(os.path.join(self.chunks_dir, f"{split}_overview.pkl"))["chunks"]
        )
        chunk_ids, local_ids, n_peaks = [], [], []
        for cid, info in enumerate(self._chunk_infos):
            n = int(info["n_samples"])
            per_sample = list(info.get("n_peaks_per_sample", [0] * n))
            if len(info["sample_indices"]) != n or len(per_sample) != n:
                raise ValueError(f"inconsistent chunk entry {info.get('file')}")
            chunk_ids.extend([cid] * n)
            local_ids.extend(range(n))
            n_peaks.extend(int(x) for x in per_sample)
        self._chunk_id_of_sample = np.asarray(chunk_ids, dtype=np.int32)
        self._local_idx_in_chunk = np.asarray(local_ids, dtype=np.int32)
        sizes = np.asarray(n_peaks, dtype=np.int32)
        if self.max_cell_peaks_num > 0:
            sizes = np.minimum(sizes, self.max_cell_peaks_num).astype(np.int32)
        self._n_peaks_per_sample = sizes
        self._sizes = sizes.astype(np.int64)
        self._indices_by_chunk = [
            np.nonzero(self._chunk_id_of_sample == cid)[0].astype(np.int64) for cid in range(len(self._chunk_infos))
        ]
        if len(self._sizes) == 0:
            raise RuntimeError(f"split {split} of {data_dir} is empty")
        self._cache = SingleFileCache()
        logger.info("%s/%s: %d cells, %d chunk files, %d peaks, %d genes",
                    data_dir, split, len(self), len(self._chunk_infos), self._n_peaks, self._num_genes)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._sizes.shape[0])

    def ordered_indices(self) -> np.ndarray:
        return grouped_ordered_indices(self.seed, self._indices_by_chunk, self._n_peaks_per_sample)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_cache"] = SingleFileCache()
        return st

    # ------------------------------------------------------------------
    def load_record(self, index: int) -> Dict[str, Any]:
        info = self._chunk_infos[int(self._chunk_id_of_sample[index])]
        return self._cache.get(os.path.join(self.chunks_dir, info["file"]))[int(self._local_idx_in_chunk[index])]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        rec = self.load_record(index)["data"]
        gex = np.asarray(rec[0], dtype=np.float32).ravel()
        if gex.shape[0] != self._num_genes:
            raise ValueError(f"cell {index}: GEX length {gex.shape[0]} != {self._num_genes}")
        peak_ids = np.asarray(rec[1], dtype=np.int64)
        celltype, barcode = str(rec[2][0]), str(rec[2][1])

        if self._unseen_peak_mask is not None and self.split == "train" and peak_ids.size > 0:
            peak_ids = peak_ids[~self._unseen_peak_mask[peak_ids]]
        if 0 < self.max_cell_peaks_num < peak_ids.size:
            peak_ids = peak_ids[: self.max_cell_peaks_num]

        return {
            "id": index,
            "src_tokens": torch.from_numpy(peak_ids + self.n_special).long(),
            "peak_chrom_id": torch.from_numpy(np.take(self._peak_chr_id, peak_ids).astype(np.int64)),
            "peak_start": torch.from_numpy(np.take(self._peak_starts, peak_ids).astype(np.int64)),
            "peak_end": torch.from_numpy(np.take(self._peak_ends, peak_ids).astype(np.int64)),
            "target": torch.from_numpy(gex),
            "meta": {"celltype": celltype, "barcode": barcode, "split": self.split},
        }

    def collater(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(samples) == 0:
            return {}
        B = len(samples)
        lengths = np.asarray([int(s["src_tokens"].numel()) for s in samples], dtype=np.int64)
        T = int(lengths.max())
        src_tokens = torch.full((B, T), self.pad_idx, dtype=torch.long)
        peak_padding_mask = torch.ones((B, T), dtype=torch.bool)
        peak_chrom_id = torch.full((B, T), self.chrom_pad_val, dtype=torch.long)
        peak_start = torch.full((B, T), self.coord_pad_val, dtype=torch.long)
        peak_end = torch.full((B, T), self.coord_pad_val, dtype=torch.long)
        for i, s in enumerate(samples):
            n = int(lengths[i])
            if n > 0:
                src_tokens[i, :n] = s["src_tokens"]
                peak_padding_mask[i, :n] = False
                peak_chrom_id[i, :n] = s["peak_chrom_id"]
                peak_start[i, :n] = s["peak_start"]
                peak_end[i, :n] = s["peak_end"]
        net_input = {
            "src_tokens": src_tokens,
            "peak_padding_mask": peak_padding_mask,
            "peak_chrom_id": peak_chrom_id,
            "peak_start": peak_start,
            "peak_end": peak_end,
            "gene_chrom_id": self._gene_chr_id_tensor,
            "gene_tss": self._gene_tss_tensor,
        }
        if self._unseen_gene_mask_tensor is not None:
            net_input["unseen_gene_mask"] = self._unseen_gene_mask_tensor
        return {
            "id": torch.LongTensor([int(s["id"]) for s in samples]),
            "net_input": net_input,
            "target": torch.stack([s["target"] for s in samples], dim=0),
            "meta": [s["meta"] for s in samples],
        }
