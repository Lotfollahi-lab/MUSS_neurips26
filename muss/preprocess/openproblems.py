"""Open Problems 2022 multiome (Kaggle "open-problems-multimodal") -> processed GEX dataset.

    python -m muss.preprocess.openproblems --raw RAW --reference REF --pbmc10k OUT/processed/pbmc/pbmc10k \
        --out OUT/processed/openproblems --weights W

* cells: the official multiome training cells (``train_multi_*``)
* GEX: the provided targets of the PBMC10K vocabulary genes (genes not measured are zero)
* peaks: the input columns on chr1-22, X, Y, M, in column order
* split: 80/10/10, cells of the held-out cell types only in valid/test (seed 42)

Reproduction note: accessible peaks of a cell on other contigs (GL*/KI* scaffolds) are kept as
peak id -1, as in the published data (about 63% of cells have one). The dataset loader turns
id -1 into token 4, the [MASK] token, placed at the coordinates of the last vocabulary peak.

Inputs: ``RAW/openproblems/{metadata.csv, train_multi_inputs.h5, train_multi_targets.h5}`` from the
Kaggle competition ``open-problems-multimodal`` (see the README for the download commands).
"""

import argparse
import os
import re
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

from .common import (
    GEXDatasetWriter, canonical_chrom, check_fasta_has_canonical_chroms, copy_reference_gene_embeddings,
    dense_gene_index, load_reference_genes, log, peaks_meta, split_positions, split_unseen_forced,
    strip_version, write_json,
)

UNSEEN_CELLTYPES = ["EryP", "MkP", "MoP", "BP"]
_PEAK_NAME = [re.compile(p, re.IGNORECASE) for p in
              (r"^(chr[^:]+):(\d+)-(\d+)$", r"^(chr[^:]+):(\d+)_(\d+)$", r"^(chr[^:]+):(\d+):(\d+)$", r"^([^:]+):(\d+)-(\d+)$")]


class PandasH5Matrix:
    """Row reader for the competition's pandas HDF5 (fixed format) matrices."""

    def __init__(self, path: str):
        import tables

        self.h5 = tables.open_file(path, mode="r")
        (name,) = list(self.h5.root._v_children.keys())
        group = self.h5.get_node(f"/{name}")
        self.values = group.block0_values
        axis0 = [x.decode() if isinstance(x, bytes) else str(x) for x in group.axis0[:]]
        axis1 = [x.decode() if isinstance(x, bytes) else str(x) for x in group.axis1[:]]
        if self.values.shape == (len(axis1), len(axis0)):
            self.row_ids, self.col_ids = axis1, axis0
        elif self.values.shape == (len(axis0), len(axis1)):
            self.row_ids, self.col_ids = axis0, axis1
        else:
            raise ValueError(f"unexpected matrix shape {self.values.shape} in {path}")

    def rows(self, start: int, stop: int) -> np.ndarray:
        return np.asarray(self.values[start:stop, :], dtype=np.float32)

    def close(self):
        self.h5.close()


def parse_peak_name(name: str) -> Tuple[str, int, int]:
    for rx in _PEAK_NAME:
        m = rx.match(str(name).strip())
        if m:
            s, e = int(m.group(2)), int(m.group(3))
            return m.group(1), min(s, e), max(s, e)
    raise ValueError(f"cannot parse peak name {name!r}")


def column_peak_vocabulary(names: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """(canonical-chromosome columns in column order, vocabulary id of every column or -1)."""
    parsed = [parse_peak_name(n) for n in names]
    chrom = [canonical_chrom(c) for c, _, _ in parsed]
    keep = np.array([c is not None for c in chrom], dtype=bool)
    vocab = pd.DataFrame({"Chromosome": [c for c in chrom if c is not None],
                          "Start": np.array([s for (_, s, _), k in zip(parsed, keep) if k], dtype=np.int64),
                          "End": np.array([e for (_, _, e), k in zip(parsed, keep) if k], dtype=np.int64)})
    ids = np.full(len(names), -1, dtype=np.int64)
    ids[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    return vocab, ids


def load_metadata(path: str, cell_ids: List[str]) -> pd.DataFrame:
    meta = pd.read_csv(path)
    meta = meta[meta["technology"].astype(str) == "multiome"].copy()
    meta["cell_id"] = meta["cell_id"].astype(str)
    meta = meta.drop_duplicates(subset=["cell_id"], keep="first").set_index("cell_id", drop=False)
    out = meta.reindex(list(cell_ids)).copy()
    missing = out["cell_id"].isna()
    if missing.any():
        log(f"[meta] {int(missing.sum())} cells missing from metadata.csv; cell type set to NA")
        out.loc[missing, "cell_id"] = out.index[missing].astype(str)
        out.loc[missing, "cell_type"] = "NA"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="raw data root (contains openproblems/)")
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
    raw = os.path.join(a.raw, "openproblems")

    inputs, targets = PandasH5Matrix(os.path.join(raw, "train_multi_inputs.h5")), PandasH5Matrix(os.path.join(raw, "train_multi_targets.h5"))
    if inputs.row_ids != targets.row_ids:
        raise RuntimeError("train_multi_inputs and train_multi_targets rows differ")
    cell_ids = [str(c) for c in inputs.row_ids]
    meta = load_metadata(os.path.join(raw, "metadata.csv"), cell_ids)
    celltypes = [str(x) for x in meta["cell_type"].tolist()]

    vocab, peak_id = column_peak_vocabulary(inputs.col_ids)
    log(f"[peaks] {len(vocab)} peaks ({len(peak_id)} columns)")
    genes = load_reference_genes(a.pbmc10k)
    dst, src = dense_gene_index([strip_version(g) for g in targets.col_ids], genes["gene_ids"])
    log(f"[genes] {len(dst)}/{genes['n_genes']} vocabulary genes measured")

    splits = split_unseen_forced(cell_ids, celltypes, 0.1, 0.1, UNSEEN_CELLTYPES, 42)
    write_json(splits, os.path.join(a.out, "split_id.json"))

    writer = GEXDatasetWriter(a.out, split_positions(splits, cell_ids))
    block = 64
    for start in range(0, len(cell_ids), block):
        stop = min(len(cell_ids), start + block)
        atac, gex = inputs.rows(start, stop), targets.rows(start, stop)
        for j in range(stop - start):
            i = start + j
            gx = np.zeros(genes["n_genes"], dtype=np.float32)
            gx[dst] = gex[j][src]
            ids = peak_id[np.flatnonzero(atac[j] != 0.0).astype(np.int32)]
            ids = (np.unique(ids) if ids.size else ids).astype(np.int32)  # keeps -1 (see module docstring)
            writer.add(i, gx, ids, celltypes[i], str(meta["cell_id"].iat[i]))
        if stop % 5120 == 0 or stop == len(cell_ids):
            log(f"[cells] {stop}/{len(cell_ids)}")
    inputs.close()
    targets.close()
    writer.finalize(genes, peaks_meta(vocab), {"dataset": "openproblems", "source": "open-problems-multimodal (multiome train)",
                                               "peak_vocab": "local_column_order", "unseen_celltypes": UNSEEN_CELLTYPES})

    if not a.skip_embeddings:
        from .alphagenome import peak_embeddings

        peak_embeddings(vocab, fasta, a.weights, os.path.join(a.out, "peak_embeddings_alphagenome.h5"))
        copy_reference_gene_embeddings(a.pbmc10k, a.out, genes["gene_ids"])
    log(f"[done] {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
