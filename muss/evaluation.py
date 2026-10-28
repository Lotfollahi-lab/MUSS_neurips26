"""Pseudobulk gene-expression metrics used for the MUSS results.

For every cell type with at least ``min_cells`` cells, predictions and measured counts
are averaged over the cells of that type (pseudobulk), optionally transformed to
log10(CPM + 1) (CPM computed over the full gene panel), restricted to a gene subset and
compared with Pearson / Spearman / CCC / MAE / RMSE / R^2. Summaries are unweighted means
over cell types.

Settings: genes are split by sequence fold (``seenseq`` / ``unseenseq`` / ``allseq``) and cells
by whether their cell type was held out of fine-tuning (``seen_celltype`` / ``unseen_celltype``).
"""

import math
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

SPACES = ("linear", "log10cpm1")


def log10_cpm_plus1(x: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.zeros_like(arr)
    return np.log10(arr / s * 1e6 + 1.0)


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if int(m.sum()) < 2:
        return float("nan")
    yt, yp = yt[m], yp[m]
    mt, mp = float(yt.mean()), float(yp.mean())
    denom = float(yt.var()) + float(yp.var()) + (mt - mp) ** 2
    if denom <= 0:
        return float("nan")
    return float(2.0 * float(np.mean((yt - mt) * (yp - mp))) / denom)


def row_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    n = int(obs.size)
    if n == 0:
        return {k: float("nan") for k in ("pearson_r", "spearman_r", "ccc", "mae", "rmse", "r2")} | {"n": 0}
    obs_s, pred_s = pd.Series(obs), pd.Series(pred)
    err = pred - obs
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return {
        "n": n,
        "pearson_r": float(obs_s.corr(pred_s, method="pearson")),
        "spearman_r": float(obs_s.corr(pred_s, method="spearman")),
        "ccc": ccc(obs, pred),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(float(np.mean(err ** 2)))),
        "r2": float(1.0 - float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def per_celltype_pseudobulk(
    pred: np.ndarray,  # [N, G] linear (count) space
    truth: np.ndarray,  # [N, G]
    celltypes: np.ndarray,  # [N]
    gene_mask: np.ndarray,  # [G]
    cell_mask: np.ndarray,  # [N]
    space: str,
    min_cells: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if int(cell_mask.sum()) == 0 or int(gene_mask.sum()) == 0:
        return out
    cts, p_keep, t_keep = celltypes[cell_mask], pred[cell_mask], truth[cell_mask]
    g_idx = np.flatnonzero(gene_mask.astype(bool))
    for ct in sorted(set(cts.tolist())):
        m = cts == ct
        n_cells = int(m.sum())
        if n_cells < int(min_cells):
            continue
        pb_pred = np.clip(p_keep[m], 0.0, None).mean(axis=0)
        pb_true = np.clip(t_keep[m], 0.0, None).mean(axis=0)
        if space == "log10cpm1":
            pb_pred, pb_true = log10_cpm_plus1(pb_pred), log10_cpm_plus1(pb_true)
        elif space != "linear":
            raise ValueError(f"unknown space {space!r}")
        pb_pred, pb_true = pb_pred[g_idx], pb_true[g_idx]
        finite = np.isfinite(pb_pred) & np.isfinite(pb_true)
        r = row_metrics(pb_true[finite], pb_pred[finite])
        r.update({"celltype": ct, "n_cells": n_cells, "n_genes": int(g_idx.size)})
        out.append(r)
    return out


def gex_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    celltypes: np.ndarray,
    splits: np.ndarray,
    unseen_gene_mask: np.ndarray,
    unseen_celltypes: Sequence[str],
    eval_splits: Sequence[str] = ("test",),
    min_cells: int = 10,
) -> pd.DataFrame:
    """Long-form per-cell-type metrics over split x gene setting x cell-type setting x space."""
    unseen_gene_mask = np.asarray(unseen_gene_mask, dtype=bool)
    unseen_ct = set(str(x) for x in unseen_celltypes)
    cell_unseen = np.asarray([str(c) in unseen_ct for c in celltypes], dtype=bool)
    rows: List[Dict[str, Any]] = []
    for sp in eval_splits:
        split_mask = splits == sp
        if int(split_mask.sum()) == 0:
            continue
        for gene_setting, gmask in (
            ("seenseq", ~unseen_gene_mask),
            ("unseenseq", unseen_gene_mask),
            ("allseq", np.ones_like(unseen_gene_mask)),
        ):
            for ct_setting, cmask in (("seen_celltype", ~cell_unseen & split_mask), ("unseen_celltype", cell_unseen & split_mask)):
                for space in SPACES:
                    for r in per_celltype_pseudobulk(pred, truth, celltypes, gmask, cmask, space, min_cells):
                        r.update({"split": sp, "gene_setting": gene_setting, "celltype_setting": ct_setting, "space": space})
                        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["split", "gene_setting", "celltype_setting", "space", "celltype", "n_cells", "n_genes",
            "pearson_r", "spearman_r", "ccc", "mae", "rmse", "r2"]
    return df[cols]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean over cell types for every (split, gene_setting, celltype_setting, space)."""
    if df.empty:
        return df
    return (
        df.groupby(["split", "gene_setting", "celltype_setting", "space"], dropna=False)
        .agg(
            n_celltypes=("celltype", "nunique"),
            n_cells_total=("n_cells", "sum"),
            pearson_r_mean=("pearson_r", "mean"),
            spearman_r_mean=("spearman_r", "mean"),
            ccc_mean=("ccc", "mean"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
        )
        .reset_index()
    )


def main_table(summary: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """The four cells reported in the paper: (seen/unseen sequence) x (seen/unseen cell type),
    log10(CPM+1) space, PCC / CCC / R^2."""
    s = summary[(summary["split"] == split) & (summary["space"] == "log10cpm1") & (summary["gene_setting"] != "allseq")]
    s = s.set_index(["gene_setting", "celltype_setting"])
    rows = []
    for g, gl in (("seenseq", "Seen sequence"), ("unseenseq", "Unseen sequence")):
        for c, cl in (("seen_celltype", "Seen cell types"), ("unseen_celltype", "Unseen cell types")):
            if (g, c) in s.index:
                r = s.loc[(g, c)]
                rows.append({"sequence": gl, "cell types": cl, "PCC": r["pearson_r_mean"], "CCC": r["ccc_mean"], "R2": r["r2_mean"],
                             "n_celltypes": int(r["n_celltypes"])})
    return pd.DataFrame(rows)
