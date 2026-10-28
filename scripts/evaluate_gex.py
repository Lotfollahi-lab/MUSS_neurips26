#!/usr/bin/env python3
"""Evaluate a fine-tuned MUSS checkpoint on the test split of a downstream dataset.

    python scripts/evaluate_gex.py --dataset bmmc --data "$MUSS_DATA/bmmc" \
        --checkpoint $MUSS_ASSETS/checkpoints/finetune/muss_bmmc.pt --out-dir results/bmmc

Run preprocessing first; --data points to the locally generated dataset.

Writes metrics_per_celltype.csv, metrics_summary.csv, main_table.csv (the four settings
reported per dataset, seen/unseen sequence x seen/unseen cell types, each with PCC / CCC / R^2
on log10(CPM+1)) and, with --save-predictions, predictions.npz.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from muss.checkpoint import load_checkpoint_state, model_config_from_checkpoint  # noqa: E402
from muss.data.common import load_pickle  # noqa: E402
from muss.evaluation import gex_metrics, main_table, summarize  # noqa: E402
from muss.models import MUSSGEXModel  # noqa: E402
from muss.data.module import DataConfig, MUSSData  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)
logger = logging.getLogger("evaluate_gex")


@torch.no_grad()
def predict(task: MUSSData, model: MUSSGEXModel, split: str, device: torch.device):
    """Predicted counts for every cell of a split (batch size 1, fixed dataset order)."""
    ds = task.dataset(split)
    itr = task.loader(split)
    preds, truths, celltypes, barcodes, index = [], [], [], [], []
    t0 = time.time()
    for i, sample in enumerate(itr):
        net = {k: v.to(device) for k, v in sample["net_input"].items() if k != "unseen_gene_mask"}
        preds.append(model(**net).float().cpu().numpy())
        truths.append(sample["target"].float().numpy())
        celltypes.extend(m["celltype"] for m in sample["meta"])
        barcodes.extend(m["barcode"] for m in sample["meta"])
        index.extend(sample["id"].tolist())
        if i % 1000 == 0:
            logger.info("%s: %d/%d cells (%.0fs)", split, i, len(ds), time.time() - t0)
    return (np.concatenate(preds), np.concatenate(truths), np.asarray(celltypes, dtype=object),
            np.asarray(barcodes, dtype=object), np.asarray(index, dtype=np.int64))


def main():
    with open(os.path.join(REPO, "configs", "datasets.json")) as f:
        datasets = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(datasets))
    ap.add_argument("--checkpoint", required=True, help="fine-tuned MUSS checkpoint (release or Lightning format)")
    ap.add_argument("--assets", default=os.environ.get("MUSS_ASSETS"), help="assets directory (default: $MUSS_ASSETS)")
    ap.add_argument("--data", help="processed dataset directory (default: <assets>/<processed path of the dataset>)")
    ap.add_argument("--sequences-bed", help="default: <assets>/reference/sequences_human.bed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()
    if not (args.data and args.sequences_bed) and not args.assets:
        ap.error("pass --assets (or set MUSS_ASSETS), or both --data and --sequences-bed")
    spec = datasets[args.dataset]
    data = args.data or os.path.join(args.assets, spec["processed"])
    bed = args.sequences_bed or os.path.join(args.assets, "reference", "sequences_human.bed")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = MUSSData(DataConfig(data=data, sequences_bed=bed, seed=1))
    task.load_embeddings()
    state = load_checkpoint_state(args.checkpoint)
    model = MUSSGEXModel.from_data(model_config_from_checkpoint(state), task)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    logger.info("loaded %s", args.checkpoint)

    pred, truth, celltypes, barcodes, index = predict(task, model, args.split, device)
    splits = np.asarray([args.split] * len(celltypes), dtype=object)
    # cell types held out of fine-tuning, as recorded by the preprocessing
    unseen_celltypes = load_pickle(os.path.join(data, "chunks", "overview.pkl"))["dataset_meta"]["unseen_celltypes"]
    per_ct = gex_metrics(pred, truth, celltypes, splits, task.unseen_gene_mask, unseen_celltypes, eval_splits=(args.split,))
    summary = summarize(per_ct)
    table = main_table(summary, split=args.split)
    per_ct.to_csv(os.path.join(args.out_dir, "metrics_per_celltype.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), index=False)
    table.to_csv(os.path.join(args.out_dir, "main_table.csv"), index=False)
    if args.save_predictions:
        np.savez_compressed(
            os.path.join(args.out_dir, "predictions.npz"),
            pred=pred.astype(np.float32), truth=truth.astype(np.float32), celltypes=celltypes.astype(str),
            cell_ids=barcodes.astype(str), cell_index=index, unseen_gene_mask=task.unseen_gene_mask,
            gene_ids=np.asarray(task.gene_ids, dtype=str),
        )
    if torch.cuda.is_available():
        logger.info("peak GPU memory allocated: %.1f GiB", torch.cuda.max_memory_allocated() / 2**30)
    print(f"\n{args.dataset} ({args.split} split, {len(celltypes)} cells), log10(CPM+1), mean over cell types:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
