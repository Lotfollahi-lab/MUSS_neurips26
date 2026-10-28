#!/usr/bin/env python3
"""Fine-tune a released MUSS encoder with PyTorch Lightning."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from muss.checkpoint import load_checkpoint_state, model_config_from_checkpoint, save_weights
from muss.data.module import DataConfig, MUSSData
from muss.models import MUSSGEXModel
from muss.training import LastCheckpoint, MUSSFineTuner


def main():
    specs = json.loads((REPO / "configs/datasets.json").read_text())
    specs = {key: value for key, value in specs.items() if not key.startswith("_")}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=specs)
    parser.add_argument("--data", required=True, help="locally preprocessed dataset")
    parser.add_argument("--assets", default=os.environ.get("MUSS_ASSETS"))
    parser.add_argument("--pretrained", help="released encoder checkpoint; defaults to MUSS_ASSETS")
    parser.add_argument("--sequences-bed", help="sequence-fold BED; defaults to MUSS_ASSETS")
    parser.add_argument("--out-dir", required=True, help="checkpoint directory")
    parser.add_argument("--log-dir", default="logs/finetune")
    parser.add_argument("--resume", help="Lightning last.ckpt from the same run")
    parser.add_argument("--devices", type=int, help="device count; defaults to the dataset's reference setting")
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--accumulate", type=int, help="cells per device per update")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-cell-peaks-num", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=4000, help="optimizer steps between recovery checkpoints")
    args = parser.parse_args()
    spec = specs[args.dataset]
    if not args.assets and (not args.sequences_bed or not (args.pretrained or args.resume)):
        parser.error("Set MUSS_ASSETS or pass --sequences-bed and --pretrained/--resume")
    devices = args.devices if args.devices is not None else spec["devices"]
    if devices < 1:
        parser.error("--devices must be positive")
    accumulation = args.accumulate
    if accumulation is None:
        if spec["effective_batch_size"] % devices:
            parser.error("The reference effective batch size must divide by --devices; otherwise set --accumulate")
        accumulation = spec["effective_batch_size"] // devices
    if accumulation < 1 or args.save_every < 1:
        parser.error("--accumulate and --save-every must be positive")
    bed = args.sequences_bed or str(Path(args.assets) / "reference/sequences_human.bed")
    source = args.resume or args.pretrained or str(Path(args.assets) / "checkpoints/pretrain/muss_pretrain.pt")
    output = Path(args.out_dir).resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error("--out-dir is not empty; use --resume or choose a new directory")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    pl.seed_everything(args.seed, workers=True)
    data = MUSSData(DataConfig(data=str(Path(args.data).resolve()), sequences_bed=str(Path(bed).resolve()),
                               seed=args.seed, max_cell_peaks_num=args.max_cell_peaks_num))
    state = load_checkpoint_state(source)
    model = MUSSGEXModel.from_data(model_config_from_checkpoint(state), data)
    if args.resume:
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_pretrained_encoder(state)
    del state
    module = MUSSFineTuner(model, data)
    checkpoint = LastCheckpoint(output / "last.ckpt", args.save_every)
    strategy = DDPStrategy(find_unused_parameters=True, broadcast_buffers=False) if devices > 1 else "auto"
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=devices, strategy=strategy, precision="32-true",
        max_steps=args.max_steps if args.max_steps is not None else spec["max_steps"],
        max_epochs=args.max_epochs if args.max_epochs is not None else spec["max_epochs"],
        accumulate_grad_batches=accumulation, use_distributed_sampler=False,
        callbacks=[checkpoint], enable_checkpointing=False, logger=CSVLogger(args.log_dir, name=args.dataset),
        default_root_dir=str(output), num_sanity_val_steps=0, log_every_n_steps=5,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(module, ckpt_path=args.resume)
    if trainer.is_global_zero:
        save_weights(model, output / "muss_finetuned.pt", trainer.global_step)


if __name__ == "__main__":
    main()
