"""Preprocessing pipelines: raw single-cell multiome data -> MUSS processed datasets.

One module per dataset, each runnable with ``python -m muss.preprocess.<module> --help``:

* :mod:`muss.preprocess.pbmc`          PBMC10K (multiome) + PBMC169K (scATAC)
* :mod:`muss.preprocess.bmmc`          NeurIPS 2021 BMMC multiome
* :mod:`muss.preprocess.openproblems`  Open Problems 2022 (Kaggle) multiome
* :mod:`muss.preprocess.hca_bone`      HCA developing bone multiome

The PBMC corpus must be built first: its PBMC10K gene vocabulary (and gene embeddings)
is reused by the three fine-tuning datasets. :mod:`muss.preprocess.alphagenome` computes
the fixed peak / gene sequence embeddings.
"""
