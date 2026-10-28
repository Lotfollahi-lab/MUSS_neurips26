import os
import pickle

import numpy as np
import pandas as pd
import pytest

from muss.preprocess import alphagenome as ag
from muss.preprocess.bmmc import local_peak_vocabulary
from muss.preprocess.common import (
    ChunkWriter, canonical_chrom, chrom_id_map, dense_gene_index, expr_ranked_lists, split_positions, split_random,
    split_stratified, split_unseen_forced,
)
from muss.preprocess import hca_bone, pbmc
from muss.preprocess.openproblems import column_peak_vocabulary
from muss.preprocess.pbmc import match_peaks, shared_names


def test_canonical_chrom_and_ids():
    assert canonical_chrom("chr1") == "chr1" and canonical_chrom("1") == "chr1"
    assert canonical_chrom("chrx") == "chrX" and canonical_chrom("MT") == "chrM"
    assert canonical_chrom("GL000194.1") is None and canonical_chrom("chrUn_KI270302v1") is None
    m = chrom_id_map(["chr2", "chrX", "KI270728.1", "chr1"])
    assert m == {"2": 2, "X": 23, "KI270728.1": 26, "1": 1}


def test_split_unseen_forced():
    ids = [f"c{i}" for i in range(200)]
    cts = ["U" if i % 5 == 0 else f"S{i % 3}" for i in range(200)]
    s = split_unseen_forced(ids, cts, 0.1, 0.1, ["U"], seed=42)
    assert s == split_unseen_forced(ids, cts, 0.1, 0.1, ["U"], seed=42)
    train, valid, test = (set(s[k]) for k in ("train_idx", "valid_idx", "test_idx"))
    assert not (train & valid) and not (train & test) and not (valid & test)
    assert train | valid | test == set(ids)
    assert all(cts[int(c[1:])] != "U" for c in train)
    assert len(train) == 160 - 16 - 16  # 160 seen cells, 10% + 10% held out


def test_split_stratified_is_per_celltype():
    ids = [f"c{i}" for i in range(300)]
    cts = [["a", "b", "c"][i % 3] for i in range(300)]
    s = split_stratified(ids, cts, 0.1, 0.1, ["c"], seed=42)
    for ct in ("a", "b"):
        n_valid = sum(cts[int(c[1:])] == ct for c in s["valid_idx"])
        assert n_valid == 10
    assert not any(cts[int(c[1:])] == "c" for c in s["train_idx"])
    assert all(s[k] == sorted(s[k], key=lambda c: int(c[1:])) for k in s)


def test_split_random():
    s = split_random(50, 0.1, 0.1, seed=1)
    assert sorted(s["train_idx"] + s["valid_idx"] + s["test_idx"]) == list(range(50))
    assert (len(s["valid_idx"]), len(s["test_idx"])) == (5, 5)
    assert s == split_random(50, 0.1, 0.1, seed=1)


def test_split_positions_validates():
    assert split_positions({"train": ["b"], "test": ["a"]}, ["a", "b"]) == {"train": [1], "test": [0]}
    with pytest.raises(KeyError):
        split_positions({"train": ["z"]}, ["a"])
    with pytest.raises(ValueError):
        split_positions({"train": ["a"], "test": ["a"]}, ["a"])


def test_dense_gene_index_first_occurrence():
    dst, src = dense_gene_index(["g2", "g1", "g2"], ["g1", "g2", "g3"])
    assert dst.tolist() == [0, 1] and src.tolist() == [1, 0]


def test_expr_ranked_lists():
    ids, vals = expr_ranked_lists(np.array([0.0, 3.2, 0.4, 7.0, 3.0], dtype=np.float32))
    assert ids.tolist() == [3, 1, 4] and vals.tolist() == [7, 3, 3]


def test_chunk_writer_respects_size(tmp_path):
    w = ChunkWriter("train", str(tmp_path), max_chunk_size_mb=1)
    payload = (np.zeros(40_000, dtype=np.float32),)
    for i in range(20):
        w.add(i, payload, n_peaks=i)
    ov = w.finalize()
    assert len(ov["chunks"]) > 1
    assert sum(c["n_samples"] for c in ov["chunks"]) == 20
    assert [i for c in ov["chunks"] for i in c["sample_indices"]] == list(range(20))
    with open(os.path.join(tmp_path, ov["chunks"][0]["file"]), "rb") as f:
        assert pickle.load(f)[0]["idx"] == 0


def test_match_peaks_rule():
    primary = pd.DataFrame({"Chromosome": ["chr1", "chr1", "chr2"], "Start": [100, 1000, 100], "End": [600, 1500, 600]})
    secondary = pd.DataFrame({"Chromosome": ["chr1", "chr1", "chr1", "chr3"], "Start": [130, 950, 100, 100],
                              "End": [620, 1560, 700, 600]})
    # 1st: both ends within 50 bp of primary 0; 2nd: start differs by exactly 50 -> no match;
    # 3rd: end differs by 100 -> no match; 4th: chromosome absent
    assert match_peaks(secondary, primary).tolist() == [0, -1, -1, -1]


def test_local_peak_vocabulary_sorting():
    names = ["chr2-50-90", "chr10-5-9", "chr1-100-200", "GL000194.1-1-5", "chr1-100-200", "chrX-1-2"]
    vocab, ids = local_peak_vocabulary(names)
    assert vocab["Chromosome"].tolist() == ["chr1", "chr10", "chr2", "chrX"]  # sorted by chromosome name
    assert ids.tolist() == [2, 1, 0, -1, 0, 3]


def test_column_peak_vocabulary_keeps_column_order():
    vocab, ids = column_peak_vocabulary(["chr2:50-90", "GL000194.1:1-5", "chr1:100-200"])
    assert vocab["Chromosome"].tolist() == ["chr2", "chr1"]
    assert ids.tolist() == [0, -1, 1]


def test_alphagenome_window_plan():
    sizes = {"chr1": 1_000_000}
    peaks = pd.DataFrame({"Chromosome": ["chr1"] * 4, "Start": [1_000, 3_000, 500_000, 600_000],
                          "End": [1_500, 3_400, 500_300, 600_000 + 200_000]})
    jobs = ag.plan_peak_windows(peaks, sizes.__getitem__)
    covered = {}
    for L, windows in jobs.items():
        assert L % ag.WINDOW_MULTIPLE_BP == 0 and ag.MIN_WINDOW_BP <= L <= ag.MAX_WINDOW_BP
        for chrom, ws, we, pieces in windows:
            assert we - ws == L and 0 <= ws and we <= sizes[chrom]
            for idx, ps, pe in pieces:
                assert ws <= ps < pe <= we
                covered[idx] = covered.get(idx, 0) + (pe - ps)
    assert covered == {0: 500, 1: 400, 2: 300, 3: 200_000}  # the long peak is cut into pieces
    assert ag.token_range(0, 100, 300, 16) == (0, 3)
    assert ag.context_window(10_000, 9_900, 9_950, 2_048) == (7_952, 10_000)


def test_incomplete_raw_data_is_an_error(tmp_path, monkeypatch):
    root = str(tmp_path)
    for name in ("s1", "s2"):
        os.makedirs(os.path.join(root, name))
        for f in hca_bone.sample_files(root, name):
            open(f, "w").close()
    monkeypatch.setattr(hca_bone, "N_SAMPLES", 2)
    assert hca_bone.discover_samples(root) == ["s1", "s2"]
    os.remove(hca_bone.sample_files(root, "s2")[1])  # one ATAC file missing
    with pytest.raises(FileNotFoundError):
        hca_bone.discover_samples(root)
    monkeypatch.setattr(hca_bone, "N_SAMPLES", 48)
    os.remove(hca_bone.sample_files(root, "s2")[0])
    os.rmdir(os.path.join(root, "s2"))
    with pytest.raises(FileNotFoundError):  # a whole sample missing
        hca_bone.discover_samples(root)


def test_incomplete_pbmc169k_is_an_error(tmp_path, monkeypatch):
    root = str(tmp_path)
    for name in ("b1", "b2"):
        os.makedirs(os.path.join(root, name))
        open(os.path.join(root, name, f"{name}.h5ad"), "w").close()
    monkeypatch.setattr(pbmc, "N_BATCHES", 2)
    assert [n for n, _ in pbmc.pbmc169k_batches(root)] == ["b1", "b2"]
    os.remove(os.path.join(root, "b2", "b2.h5ad"))
    with pytest.raises(FileNotFoundError):  # a batch directory without its h5ad
        pbmc.pbmc169k_batches(root)
    os.rmdir(os.path.join(root, "b2"))
    with pytest.raises(FileNotFoundError):  # a whole batch missing
        pbmc.pbmc169k_batches(root)


def test_shared_names_pickle_each_name_once():
    values = ["".join(("B_", "CELLS")), "NK_CELLS", "".join(("B_", "CELLS"))]
    assert values[0] is not values[2] and pickle.dumps(values).count(b"B_CELLS") == 2
    names = shared_names(pd.Series(values))
    assert names == values and names[0] is names[2]
    assert pickle.dumps(names).count(b"B_CELLS") == 1
