"""Lightning fine-tuning with the released model's loss and update rules."""

from dataclasses import asdict

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from .data.module import CellSampler, CellView
from .optim import MUSSAdam, WarmupInverseSqrt


def poisson_loss(prediction, target, unseen_gene_mask, training):
    rate = prediction.float().clamp_min(1e-8)
    target = target.float()
    seen = ~unseen_gene_mask.bool()
    if training:
        target = target[:, seen]
        if rate.shape != target.shape:
            raise ValueError("Training predictions must contain only seen genes")
        return F.poisson_nll_loss(rate, target, log_input=False, full=False, reduction="none").sum(-1).mean()
    if rate.shape != target.shape:
        raise ValueError("Evaluation predictions must contain every gene")
    loss = F.poisson_nll_loss(rate, target, log_input=False, full=False, reduction="none")
    return (loss * seen.to(loss.dtype)).sum(-1).mean()


def pearson_per_cell(prediction, target, eps=1e-8):
    x = prediction - prediction.mean(dim=1, keepdim=True)
    y = target - target.mean(dim=1, keepdim=True)
    denominator = x.square().sum(1).sqrt().clamp_min(eps) * y.square().sum(1).sqrt().clamp_min(eps)
    return torch.nan_to_num((x * y).sum(1) / denominator, nan=0.0, posinf=0.0, neginf=0.0)


class MUSSFineTuner(pl.LightningModule):
    def __init__(self, model, data, learning_rate=1e-4, warmup=100):
        super().__init__()
        self.model, self.data = model.train(), data
        self.learning_rate, self.warmup = learning_rate, warmup
        self._seeded_step = -1
        self._grad_factor = 1.0

    def forward(self, **inputs):
        return self.model(**inputs)

    def train_dataloader(self):
        dataset = self.data.dataset("train")
        view = CellView(dataset)
        sampler = CellSampler(dataset, self.trainer.world_size, self.global_rank,
                              self.trainer.accumulate_grad_batches)
        return StatefulDataLoader(view, batch_size=1, sampler=sampler, collate_fn=view.collate,
                                  num_workers=0, pin_memory=self.device.type == "cuda")

    def val_dataloader(self):
        dataset = self.data.dataset("valid")
        view = CellView(dataset)
        sampler = CellSampler(dataset, self.trainer.world_size, self.global_rank)
        return DataLoader(view, batch_size=1, sampler=sampler, collate_fn=view.collate, num_workers=0)

    def on_train_batch_start(self, batch, batch_idx):
        if self._seeded_step != self.global_step:
            torch.manual_seed(self.data.cfg.seed + self.global_step)
            self._seeded_step = self.global_step
        self._grad_factor = batch["grad_factor"]

    def training_step(self, batch, batch_idx):
        prediction = self.model(**batch["net_input"])
        loss = poisson_loss(prediction, batch["target"], batch["net_input"]["unseen_gene_mask"], True)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, batch_size=1)
        if batch["is_dummy"]:
            loss = loss * 0
        # Undo Lightning's fixed divisor; normalize the accumulated gradients once below.
        return loss * self.trainer.accumulate_grad_batches

    def on_before_optimizer_step(self, optimizer):
        for parameter in self.model.parameters():
            if parameter.grad is None and self.trainer.world_size > 1 and parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
            if parameter.grad is not None:
                parameter.grad.mul_(self._grad_factor)

    def configure_optimizers(self):
        optimizer = MUSSAdam(self.model.parameters(), lr=self.learning_rate)
        scheduler = WarmupInverseSqrt(optimizer, self.warmup)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_validation_epoch_start(self):
        self._validation_totals = torch.zeros(5, dtype=torch.float64, device=self.device)

    def validation_step(self, batch, batch_idx):
        rate = self.model(**batch["net_input"]).float().clamp_min(1e-8)
        target = batch["target"].float()
        unseen = batch["net_input"]["unseen_gene_mask"].bool()
        loss = poisson_loss(rate, target, unseen, False)
        values = torch.stack([loss, pearson_per_cell(rate, target).sum(),
                              pearson_per_cell(rate[:, ~unseen], target[:, ~unseen]).sum(),
                              pearson_per_cell(rate[:, unseen], target[:, unseen]).sum(),
                              loss.new_tensor(1.0)])
        if not batch["is_dummy"]:
            self._validation_totals += values.double()

    def on_validation_epoch_end(self):
        totals = self.trainer.strategy.reduce(self._validation_totals, reduce_op="sum")
        names = ("val/loss", "val/pearson", "val/pearson_seen_gene", "val/pearson_unseen_gene")
        for name, value in zip(names, totals[:4] / totals[4].clamp_min(1)):
            self.log(name, value, sync_dist=False, prog_bar=name == "val/loss")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["muss"] = {
            "model_config": asdict(self.model.cfg), "data_config": asdict(self.data.cfg),
            "world_size": self.trainer.world_size, "accumulation": self.trainer.accumulate_grad_batches,
            "learning_rate": self.learning_rate, "warmup": self.warmup,
        }

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint["muss"]
        current = {"model_config": asdict(self.model.cfg), "data_config": asdict(self.data.cfg),
                   "world_size": self.trainer.world_size, "accumulation": self.trainer.accumulate_grad_batches,
                   "learning_rate": self.learning_rate, "warmup": self.warmup}
        if saved != current:
            changed = [key for key in current if saved.get(key) != current[key]]
            raise ValueError(f"Resume settings changed: {changed}. Start a separate run instead.")


class LastCheckpoint(pl.Callback):
    """Save at update boundaries, before Lightning releases the loader's resume state."""

    def __init__(self, path, every_n_updates=4000):
        self.path = str(path)
        self.every_n_updates = every_n_updates

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_before = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self._step_before:
            return
        last_batch = batch_idx + 1 == trainer.num_training_batches
        last_step = trainer.max_steps > 0 and step >= trainer.max_steps
        if last_batch or last_step or step % self.every_n_updates == 0:
            trainer.save_checkpoint(self.path)
