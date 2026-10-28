"""MUSS encoder and gene-expression head, compatible with the released weights."""

import logging
from os import PathLike
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..checkpoint import load_checkpoint_state
from .layers import PeakGeneDistanceBias, PeakGeneLayer, PeakSelfAttnLayer, PoissonCountHead, compute_peak_gene_dist

logger = logging.getLogger(__name__)

LAYER_TYPES = ("peak_self_attn", "isab")


@dataclass
class MUSSConfig:
    d_model: int = 2048
    d_input: int = 3072
    layer_types: str = "peak_self_attn,isab,isab"
    num_heads: int = 16
    dropout: float = 0.1
    ffn_mult: int = 2
    num_rbf_kernels: int = 32
    rpe_hidden_dim: int = 64
    dist_mask_value: float = -35.0


def parse_layer_types(value: str) -> List[str]:
    types = [t.strip() for t in str(value).split(",") if t.strip()]
    if not types or any(t not in LAYER_TYPES for t in types):
        raise ValueError(f"layer_types must be a comma-separated list of {LAYER_TYPES}, got {value!r}")
    if types.count("isab") == 0:
        raise ValueError("MUSS needs at least one isab layer")
    return types


class MUSSEncoder(nn.Module):
    """Encoder with parameter names matching the released checkpoints."""

    def __init__(self, cfg: MUSSConfig, dictionary, pretrained_peak_emb: Tensor, pretrained_gene_emb: Tensor):
        super().__init__()
        self.cfg = cfg
        self.pad_idx = int(dictionary.pad())
        self.n_special = int(dictionary.n_special)
        self.num_peaks = int(pretrained_peak_emb.shape[0])
        self.num_genes = int(pretrained_gene_emb.shape[0])
        if len(dictionary) != self.n_special + self.num_peaks:
            raise ValueError(f"vocab size {len(dictionary)} != n_special + n_peaks ({self.n_special} + {self.num_peaks})")
        D_in, D = int(cfg.d_input), int(cfg.d_model)
        for name, emb in (("peak", pretrained_peak_emb), ("gene", pretrained_gene_emb)):
            if emb.ndim != 2 or emb.shape[1] != D_in:
                raise ValueError(f"{name} embedding must be [N, d_input={D_in}], got {tuple(emb.shape)}")

        # NOTE: module construction order determines random initialization; keep it.
        self.special_embed = nn.Embedding(self.n_special, D_in, padding_idx=self.pad_idx)
        nn.init.trunc_normal_(self.special_embed.weight, std=0.02)
        with torch.no_grad():
            self.special_embed.weight[self.pad_idx].fill_(0.0)
        self.register_buffer("pretrained_peak_emb", pretrained_peak_emb.to(torch.float16), persistent=False)
        self.input_ffn = self._ffn(D_in, D)

        self.encoder_layer_types = parse_layer_types(cfg.layer_types)
        self.n_isab_layers = self.encoder_layer_types.count("isab")

        self.gene_input_ffn = self._ffn(D_in, D)
        self.register_buffer("pretrained_gene_emb", pretrained_gene_emb.to(torch.float16), persistent=False)
        self.dist_bias_module = PeakGeneDistanceBias(
            num_heads=cfg.num_heads,
            n_layers=self.n_isab_layers,
            num_kernels=cfg.num_rbf_kernels,
            hidden_dim=cfg.rpe_hidden_dim,
            mask_value=cfg.dist_mask_value,
        )

        self.encoder_layers = nn.ModuleList()
        for t in self.encoder_layer_types:
            if t == "peak_self_attn":
                self.encoder_layers.append(PeakSelfAttnLayer(D, cfg.num_heads, dropout=cfg.dropout, ffn_mult=cfg.ffn_mult))
            else:
                self.encoder_layers.append(PeakGeneLayer(D, cfg.num_heads, dropout=cfg.dropout, ffn_mult=cfg.ffn_mult))

    def load_pretrained_encoder(self, checkpoint) -> None:
        """Load every encoder tensor; leave the downstream prediction head untouched."""
        state = (load_checkpoint_state(checkpoint) if isinstance(checkpoint, (str, bytes, PathLike)) else checkpoint)["model"]
        own = self.state_dict()
        encoder_state = {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIXES)}
        for k, v in encoder_state.items():
            if k not in own or tuple(v.shape) != tuple(own[k].shape):
                raise ValueError(f"pretrained encoder key {k} is missing or has a different shape")
        missing, _ = self.load_state_dict(encoder_state, strict=False)
        not_loaded = [k for k in missing if not k.startswith("pred_head.")]
        if not_loaded:
            raise RuntimeError(f"incomplete encoder load: missing {not_loaded}")
        logger.info("loaded %d encoder tensors", len(encoder_state))

    @staticmethod
    def _ffn(d_in: int, d_out: int) -> nn.Sequential:
        ffn = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        nn.init.zeros_(ffn[1].bias)
        nn.init.zeros_(ffn[3].bias)
        return ffn

    @classmethod
    def from_data(cls, cfg, data):
        data.load_embeddings()
        return cls(
            cfg,
            data.dictionary,
            pretrained_peak_emb=torch.from_numpy(data.peak_emb_matrix),
            pretrained_gene_emb=torch.from_numpy(data.gene_emb_matrix),
        )

    # ------------------------------------------------------------------
    def _build_peak_repr(self, src_tokens: Tensor, peak_padding_mask: Tensor) -> Tensor:
        src_long = src_tokens.to(torch.long)
        is_peak = src_long >= self.n_special
        special_idx = src_long.clamp(min=0, max=self.n_special - 1)
        peak_id = (src_long - self.n_special).clamp(min=0, max=self.num_peaks - 1)

        special_emb = self.special_embed(special_idx)
        with torch.no_grad():
            peak_emb_fixed = F.embedding(peak_id, self.pretrained_peak_emb)
        peak_emb_fixed = peak_emb_fixed.to(dtype=special_emb.dtype)

        x_emb = torch.where(is_peak.unsqueeze(-1), peak_emb_fixed, special_emb)
        x_emb = x_emb.masked_fill(peak_padding_mask.unsqueeze(-1), 0.0)
        return self.input_ffn(x_emb.to(self.special_embed.weight.dtype))

    def _build_gene_repr(self, batch_size: int, device: torch.device, gene_idx: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            emb = self.pretrained_gene_emb
            if gene_idx is not None:
                emb = emb.index_select(0, gene_idx.to(device=emb.device, dtype=torch.long))
            gene_emb_fixed = emb.unsqueeze(0).expand(batch_size, -1, -1)
            gene_emb_fixed = gene_emb_fixed.to(device=device, dtype=self.special_embed.weight.dtype)
        return self.gene_input_ffn(gene_emb_fixed)

    def encode(
        self,
        src_tokens: Tensor,  # [B, T] peak ids + n_special (or special tokens)
        peak_padding_mask: Tensor,  # [B, T], True = padding
        peak_chrom_id: Tensor,
        peak_start: Tensor,
        peak_end: Tensor,
        gene_chrom_id: Tensor,  # [G]
        gene_tss: Tensor,  # [G]
        gene_idx: Optional[Tensor] = None,  # optional subset of genes to model
    ) -> Tuple[Tensor, Tensor]:
        """Returns (peak_repr [B, T, D], gene_repr [B, G', D])."""
        peak_padding_mask = peak_padding_mask.to(torch.bool)
        if gene_idx is not None:
            gene_chrom_id = gene_chrom_id.index_select(0, gene_idx.to(gene_chrom_id.device))
            gene_tss = gene_tss.index_select(0, gene_idx.to(gene_tss.device))
        n_genes = int(gene_tss.numel())
        B, T = src_tokens.shape
        device = src_tokens.device

        if T == 0:  # a cell without peaks: genes only see their own embedding
            gene_repr = self._build_gene_repr(B, device, gene_idx)
            return src_tokens.new_zeros((B, 0, self.cfg.d_model), dtype=gene_repr.dtype), gene_repr

        peak_gene_dist = compute_peak_gene_dist(peak_chrom_id, peak_start, peak_end, gene_chrom_id, gene_tss, peak_padding_mask)
        # never let a row be fully masked (keeps the softmax finite)
        attn_padding_mask = peak_padding_mask
        all_pad = attn_padding_mask.all(dim=1)
        if bool(all_pad.any()):
            attn_padding_mask = attn_padding_mask.clone()
            attn_padding_mask[all_pad, 0] = False

        x = self._build_peak_repr(src_tokens, peak_padding_mask)
        attn_bias = self.dist_bias_module(peak_gene_dist, peak_padding_mask=attn_padding_mask).view(
            B, self.n_isab_layers, self.cfg.num_heads, T, n_genes
        )
        gene_repr = self._build_gene_repr(B, device, gene_idx)

        isab_i = 0
        for layer_type, layer in zip(self.encoder_layer_types, self.encoder_layers):
            if layer_type == "peak_self_attn":
                x = layer(x, key_padding_mask=attn_padding_mask)
            else:
                x, gene_repr = layer(x, gene_repr, peak_padding_mask=attn_padding_mask, dist_bias=attn_bias[:, isab_i])
                isab_i += 1
        return x, gene_repr


ENCODER_PREFIXES = ("special_embed.", "input_ffn.", "gene_input_ffn.", "encoder_layers.", "dist_bias_module.")


class MUSSGEXModel(MUSSEncoder):
    """MUSS encoder + gene-wise Poisson head predicting the raw count of every gene."""

    def __init__(self, cfg, dictionary, pretrained_peak_emb, pretrained_gene_emb):
        super().__init__(cfg, dictionary, pretrained_peak_emb, pretrained_gene_emb)
        if self.encoder_layer_types[-1] != "isab":
            raise ValueError("muss_gex reads genes from the last layer, which must be an isab layer")
        self.pred_head = PoissonCountHead(cfg.d_model)

    def forward(
        self,
        src_tokens,
        peak_padding_mask,
        peak_chrom_id,
        peak_start,
        peak_end,
        gene_chrom_id,
        gene_tss,
        unseen_gene_mask: Optional[Tensor] = None,
        **unused,
    ) -> Tensor:
        """Predicted Poisson rate per gene: [B, G] in eval mode.

        In training mode, genes flagged in ``unseen_gene_mask`` (TSS inside a held-out
        sequence fold) are removed from the model entirely and the output is [B, G_seen].
        """
        gene_idx = None
        if unseen_gene_mask is not None and self.training:
            um = unseen_gene_mask.to(dtype=torch.bool).view(-1)
            if int(um.numel()) != self.num_genes:
                raise ValueError(f"unseen_gene_mask has {int(um.numel())} entries, expected {self.num_genes}")
            gene_idx = torch.nonzero(~um, as_tuple=False).view(-1).contiguous()
        _, gene_repr = self.encode(
            src_tokens, peak_padding_mask, peak_chrom_id, peak_start, peak_end, gene_chrom_id, gene_tss, gene_idx=gene_idx
        )
        return self.pred_head(gene_repr)
