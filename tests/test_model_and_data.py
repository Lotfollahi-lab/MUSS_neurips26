from dataclasses import asdict

import numpy as np
import pytest
import torch

from muss.checkpoint import load_checkpoint_state, model_config_from_checkpoint, save_weights
from muss.data.module import DataConfig, MUSSData
from muss.data.unseen_seq import load_holdout_windows
from muss.models import MUSSConfig, MUSSEncoder, MUSSGEXModel
from muss.models.layers import PeakGeneDistanceBias, compute_peak_gene_dist
from muss.training import poisson_loss

TINY = dict(d_model=32, d_input=16, num_heads=2, num_rbf_kernels=4, rpe_hidden_dim=8, dropout=0.0)


@pytest.fixture()
def data(corpus):
    return MUSSData(DataConfig(data=corpus["pbmc10k"], sequences_bed=corpus["bed"]))


def test_unseen_masks(data):
    assert data.unseen_gene_mask.tolist() == [True, True, False, False, False, False, False]
    assert int(data.unseen_peak_mask.sum()) == 27


def test_holdout_folds_must_exist(corpus):
    for folds in ({"fold9"}, {"fold3", "fodl4"}):
        with pytest.raises(ValueError):
            load_holdout_windows(corpus["bed"], folds)


def test_gex_dataset_items(data):
    ds = data.dataset("train")
    assert [len(data.dataset(s)) for s in ("train", "valid", "test")] == [10, 7, 7]
    item = ds[0]
    assert item["target"].shape == (7,)
    assert not data.unseen_peak_mask[item["src_tokens"].numpy() - 5].any()
    batch = ds.collater([ds[0], ds[1]])["net_input"]
    assert batch["src_tokens"].shape == batch["peak_padding_mask"].shape == batch["peak_start"].shape
    assert batch["gene_tss"].shape == (7,)
    order = ds.ordered_indices()
    assert sorted(order.tolist()) == list(range(len(ds)))
    assert np.array_equal(order, ds.ordered_indices())


def test_forward_and_loss(data):
    model = MUSSGEXModel.from_data(MUSSConfig(**TINY), data)
    ds = data.dataset("train")
    sample = ds.collater([ds[i] for i in range(3)])
    unseen = sample["net_input"]["unseen_gene_mask"]
    model.train()
    pred = model(**sample["net_input"])
    assert pred.shape == (3, 5)
    loss = poisson_loss(pred, sample["target"], unseen, True)
    loss.backward()
    assert model.pred_head.softmax_output.weight.grad is not None
    model.eval()
    with torch.no_grad():
        pred = model(**sample["net_input"])
    assert pred.shape == (3, 7) and bool((pred >= 0).all())
    expected = torch.nn.functional.poisson_nll_loss(
        pred[:, ~unseen], sample["target"][:, ~unseen], log_input=False, reduction="none"
    ).sum(-1).mean()
    torch.testing.assert_close(poisson_loss(pred, sample["target"], unseen, False), expected)


def test_encoder_loading_and_weight_export(data, tmp_path):
    encoder = MUSSEncoder.from_data(MUSSConfig(**TINY), data)
    state = {"model": encoder.state_dict(), "cfg": {"model": asdict(encoder.cfg)}}
    state["model"]["lm_bias"] = torch.zeros(3)
    model = MUSSGEXModel.from_data(encoder.cfg, data)
    head = {k: v.clone() for k, v in model.pred_head.state_dict().items()}
    model.load_pretrained_encoder(state)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for key, value in head.items():
        torch.testing.assert_close(model.pred_head.state_dict()[key], value, rtol=0, atol=0)
    save_weights(model, tmp_path / "weights.pt", 7)
    saved = load_checkpoint_state(tmp_path / "weights.pt")
    assert saved["meta"]["num_updates"] == 7
    assert model_config_from_checkpoint(saved) == model.cfg
    assert not any("pretrained_" in key for key in saved["model"])
    del state["model"]["special_embed.weight"]
    with pytest.raises(RuntimeError, match="incomplete encoder"):
        model.load_pretrained_encoder(state)


def test_distance_bias_is_bidirectional_and_masks_only_trans_pairs():
    # gene 0: chr1, TSS 1,000; gene 1: chr2, TSS 1,000. Peaks: downstream and upstream of the TSS
    # (closest boundary 100 bp away), over the TSS, on chr2, padding.
    chrom = torch.tensor([[1, 1, 1, 2, 1]])
    start = torch.tensor([[1_100, 700, 950, 1_100, 0]])
    end = torch.tensor([[1_300, 900, 1_020, 1_300, 0]])
    pad = torch.tensor([[False, False, False, False, True]])
    dist = compute_peak_gene_dist(chrom, start, end, torch.tensor([1, 2]), torch.tensor([1_000, 1_000]), pad)
    inf = float("inf")
    assert dist.tolist() == [[[100, inf], [100, inf], [20, inf], [inf, 100], [inf, inf]]]
    torch.manual_seed(0)
    module = PeakGeneDistanceBias(num_heads=2, n_layers=3, num_kernels=4, hidden_dim=8, mask_value=-35.0)
    bias = module(dist, pad)  # [B, heads x layers, peaks, genes]
    assert bias.shape == (1, 6, 5, 2)
    assert torch.equal(bias[0, :, 0], bias[0, :, 1])  # the same bias up- and downstream of the TSS
    assert bool((bias[0, :, :3, 0] != -35.0).all()) and bool((bias[0, :, 3, 1] != -35.0).all())  # cis pairs
    assert bool((bias[0, :, 3, 0] == -35.0).all()) and bool((bias[0, :, :3, 1] == -35.0).all())  # trans pairs
    assert bool((bias[0, :, 4] == 1e-4).all())  # padded peak
    with pytest.raises(ValueError):  # signed distances are rejected
        module(torch.where(torch.isinf(dist), dist, -dist), pad)
