import lightning.pytorch as pl
import pytest
import torch

from muss.checkpoint import load_checkpoint_state
from muss.data.module import CellSampler, DataConfig, MUSSData
from muss.models import MUSSConfig, MUSSGEXModel
from muss.optim import MUSSAdam, WarmupInverseSqrt
from muss.training import LastCheckpoint, MUSSFineTuner


def make_module(corpus):
    torch.manual_seed(1)
    data = MUSSData(DataConfig(data=corpus["pbmc10k"], sequences_bed=corpus["bed"]))
    config = MUSSConfig(d_model=16, d_input=16, num_heads=2, num_rbf_kernels=4, rpe_hidden_dim=8, dropout=0.1)
    return MUSSFineTuner(MUSSGEXModel.from_data(config, data), data)


def trainer(path, steps, **kwargs):
    return pl.Trainer(accelerator="cpu", devices=1, precision="32-true", max_steps=steps,
                      max_epochs=-1, accumulate_grad_batches=4, use_distributed_sampler=False,
                      logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      callbacks=[LastCheckpoint(path / "last.ckpt")],
                      enable_model_summary=False, num_sanity_val_steps=0,
                      default_root_dir=str(path), **kwargs)


def test_optimizer_warmup_and_small_gradients():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MUSSAdam([p]); scheduler = WarmupInverseSqrt(optimizer)
    p.grad = torch.tensor([1e-7])
    optimizer.step(); scheduler.step()
    assert p.item() == 1.0 and optimizer.param_groups[0]["lr"] == 1e-6
    optimizer.step(); scheduler.step()
    # Epsilon is added to the uncorrected second moment.
    mean = (1 - 0.9 ** 2) * 1e-7
    variance = (1 - 0.98 ** 2) * 1e-14
    expected = 1 - 1e-6 * (1 - 0.98 ** 2) ** 0.5 / (1 - 0.9 ** 2) * mean / (variance ** 0.5 + 1e-6)
    assert p.item() == torch.tensor(expected).item()
    for _ in range(100):
        optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]["lr"] == (1e-4 * 100 ** 0.5) * 102 ** -0.5


def test_sampler_shards_and_padding():
    class Data:
        def ordered_indices(self):
            return list(range(11))
    ranks = [list(CellSampler(Data(), 2, rank, 4)) for rank in range(2)]
    assert [key[0] for key in ranks[0]] == [0, 2, 4, 6, 8, 10]
    assert [key[0] for key in ranks[1]] == [1, 3, 5, 7, 9, 0]
    assert ranks[1][-1][1]
    assert ranks[0][-1][2] == 1.0  # original final-group normalization: world / (2 * 1)


@pytest.mark.parametrize("stop", [1, 2, 3, 4])
def test_lightning_resume_keeps_data_position_and_updates(corpus, tmp_path, stop):
    full = make_module(corpus)
    trainer(tmp_path / "full", 8).fit(full)
    partial = make_module(corpus)
    first = trainer(tmp_path / "partial", stop)
    first.fit(partial)
    checkpoint = tmp_path / "partial" / "last.ckpt"
    resumed = make_module(corpus)
    last = trainer(tmp_path / "resumed", 8)
    last.fit(resumed, ckpt_path=checkpoint)
    assert last.global_step == 8
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
    state = load_checkpoint_state(checkpoint)
    for key, value in partial.model.state_dict().items():
        torch.testing.assert_close(state["model"][key], value, rtol=0, atol=0)


def test_resume_rejects_changed_seed(corpus, tmp_path):
    module = make_module(corpus)
    first = trainer(tmp_path / "first", 1)
    first.fit(module)
    checkpoint = tmp_path / "first" / "last.ckpt"
    changed = make_module(corpus)
    changed.data.cfg.seed = 2
    with pytest.raises(ValueError, match="Resume settings changed"):
        trainer(tmp_path / "changed", 2).fit(changed, ckpt_path=checkpoint)
