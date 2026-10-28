import os

import numpy as np

from muss.evaluation import ccc, gex_metrics, log10_cpm_plus1, main_table, summarize


def test_log10_cpm_plus1():
    out = log10_cpm_plus1(np.array([1.0, 3.0, 0.0]))
    np.testing.assert_allclose(out, np.log10(np.array([0.25e6, 0.75e6, 0.0]) + 1.0))
    assert np.all(log10_cpm_plus1(np.zeros(3)) == 0)


def test_ccc():
    x = np.arange(10, dtype=float)
    assert abs(ccc(x, x) - 1.0) < 1e-12
    assert ccc(x, x + 5) < 1.0  # a shift lowers the concordance but not the correlation


def test_gex_metrics_perfect_prediction():
    rng = np.random.RandomState(0)
    truth = rng.poisson(3.0, size=(60, 30)).astype(float)
    celltypes = np.array(["a", "b", "u"] * 20, dtype=object)
    splits = np.array(["test"] * 60, dtype=object)
    unseen_gene = np.zeros(30, dtype=bool)
    unseen_gene[:8] = True
    per_ct = gex_metrics(truth.copy(), truth, celltypes, splits, unseen_gene, ["u"])
    assert set(per_ct["celltype"]) == {"a", "b", "u"}
    np.testing.assert_allclose(per_ct["pearson_r"], 1.0)
    np.testing.assert_allclose(per_ct["r2"], 1.0)
    table = main_table(summarize(per_ct), split="test")
    assert len(table) == 4
    assert np.allclose(table[["PCC", "CCC", "R2"]].to_numpy(), 1.0)


def test_min_cells_filter():
    truth = np.arange(60, dtype=float).reshape(12, 5) + 1.0
    celltypes = np.array(["a"] * 10 + ["rare"] * 2, dtype=object)
    per_ct = gex_metrics(truth, truth, celltypes, np.array(["test"] * 12, dtype=object), np.zeros(5, bool), [])
    assert set(per_ct["celltype"]) == {"a"}


def test_cell_clustering_is_deterministic():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cell_clustering", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "cell_clustering.py"))
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    rng = np.random.RandomState(0)
    centers = rng.normal(scale=10.0, size=(3, 60))
    labels = np.repeat(np.array(["a", "b", "c"], dtype=object), 50)
    emb = (centers[np.repeat(np.arange(3), 50)] + rng.normal(size=(150, 60))).astype(np.float32)
    runs = cc.cluster(emb, labels)
    assert runs["seed"].tolist() == cc.KMEANS_SEEDS
    assert bool((runs[["ARI", "FMI", "NMI", "AMI"]] > 0.999).all().all())  # well-separated clusters are recovered
    assert runs.equals(cc.cluster(emb, labels))
