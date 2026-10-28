"""Building blocks of the MUSS encoder.

These layers are numerically identical to the ones used to fine-tune and evaluate the
released checkpoints. Parameter names are part of the checkpoint format; do not rename them.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ----------------------------------------------------------------------------
# Peak-gene distance -> attention bias
# ----------------------------------------------------------------------------
class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion of log1p(distance), for distances >= 0."""

    def __init__(self, num_kernels: int):
        super().__init__()
        self.num_kernels = num_kernels
        self.mu = nn.Parameter(torch.linspace(0.0, 5.0, num_kernels))
        self.gamma = nn.Parameter(torch.ones(num_kernels))

    def forward(self, dist: Tensor) -> Tensor:
        x = torch.log1p(dist).unsqueeze(-1)
        gamma = F.softplus(self.gamma)
        diff = x - self.mu
        return torch.exp(-gamma * diff * diff)


class NonLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PeakGeneDistanceBias(nn.Module):
    """Maps the peak-gene distance matrix to one additive attention bias per head and ISAB layer.

    The distances are unsigned (``compute_peak_gene_dist``), so a peak gets the same bias at
    the same distance on either side of a gene's TSS: the gene<->peak attention is
    bidirectional. Only pairs on different chromosomes (and padded peaks) are masked.

        cis pairs:        bias = RBF_MLP(log1p(d)) - softplus(abc_lambda) * log1p(d)
        d = +inf:         bias = mask_value  (pairs on different chromosomes; compute_peak_gene_dist
                          also gives padded peaks +inf)
        peaks flagged in ``peak_padding_mask``: bias = 1e-4

    Every head of every ISAB layer has its own bias; the decay rates start at
    softplus(abc_lambda) = 1.
    """

    def __init__(self, num_heads: int, n_layers: int, num_kernels: int, hidden_dim: int, mask_value: float):
        super().__init__()
        self.num_heads = int(num_heads)
        self.n_layers = int(n_layers)
        self.mask_value = float(mask_value)
        self.rpe_heads = self.num_heads * self.n_layers

        self.rbf = GaussianRBF(int(num_kernels))
        self.rbf_proj = NonLinear(int(num_kernels), self.rpe_heads, hidden_dim=hidden_dim)
        self.abc_lambda = nn.Parameter(torch.full((self.rpe_heads,), math.log(math.expm1(1.0)), dtype=torch.float32))

    def forward(
        self,
        dist: Tensor,  # [B, P, G], >= 0; +inf marks trans pairs and padded peaks
        peak_padding_mask: Tensor,  # [B, P]
    ) -> Tensor:
        dist = dist.to(torch.float32)
        if bool(dist.lt(0).any()):
            raise ValueError("peak-gene distances must be unsigned (>= 0, +inf for trans pairs)")
        trans_mask = torch.isinf(dist)
        dist = dist.masked_fill(trans_mask, 0.0)  # placeholder, the bias of these pairs is masked below

        rbf_feat = self.rbf(dist).to(dtype=self.rbf_proj.fc1.weight.dtype)
        bias = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2).contiguous()  # [B, rpe_heads, P, G]

        lam = F.softplus(self.abc_lambda).view(1, -1, 1, 1)
        decay = torch.log1p(dist).unsqueeze(1).to(dtype=bias.dtype)
        bias = bias - lam * decay

        bias = bias.masked_fill(trans_mask.unsqueeze(1), self.mask_value)
        return bias.masked_fill(peak_padding_mask.unsqueeze(1).unsqueeze(-1), 1e-4)


def compute_peak_gene_dist(
    peak_chrom_id: Tensor,  # [B, T]
    peak_start: Tensor,  # [B, T]
    peak_end: Tensor,  # [B, T]
    gene_chrom_id: Tensor,  # [G]
    gene_tss: Tensor,  # [G]
    peak_padding_mask: Tensor,  # [B, T], True = padding
) -> Tensor:
    """Unsigned distance (bp) between each gene's TSS and the closest boundary of each peak.

    ``min(|start - tss|, |end - tss|)``, also when the TSS lies inside the peak; it does not
    depend on the side of the TSS the peak lies on. Pairs on different chromosomes and
    padded peaks get +inf.
    """
    B, T = peak_chrom_id.shape
    device = peak_chrom_id.device
    G = int(gene_chrom_id.numel())
    gene_chr = gene_chrom_id.to(device=device, dtype=torch.long).view(1, 1, G)
    tss = gene_tss.to(device=device, dtype=torch.long).view(1, 1, G)

    d_start = peak_start.to(device=device, dtype=torch.long).unsqueeze(-1) - tss
    d_end = peak_end.to(device=device, dtype=torch.long).unsqueeze(-1) - tss
    dist = torch.minimum(d_start.abs(), d_end.abs()).to(torch.float32)

    same_chr = peak_chrom_id.to(device=device, dtype=torch.long).unsqueeze(-1).eq(gene_chr)
    invalid = torch.full_like(dist, float("inf"))
    dist = torch.where(same_chr, dist, invalid)
    dist = torch.where(peak_padding_mask.to(device=device, dtype=torch.bool).unsqueeze(-1), invalid, dist)
    return dist


# ----------------------------------------------------------------------------
# Peak self-attention layer (pre-LN, rotary-style position encoding over the peak order)
#
# NOTE: this is not standard RoPE. _rotate_half pairs adjacent dimensions (2i, 2i+1) while
# cos / sin use the concatenated ("half") frequency layout, so the two members of a pair use
# different frequencies: the transform is an absolute-position encoding that neither
# preserves norms nor makes q.k depend only on the offset. The released checkpoints were
# trained with it; do not replace it by standard RoPE.
# ----------------------------------------------------------------------------
def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_1d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"RoPE dim must be positive and even, got {dim}")
        self.dim = int(dim)
        self.base = int(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype=dtype)[None, None], emb.sin().to(dtype=dtype)[None, None]


class PeakSelfAttnLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ffn_mult: int = 2, mask_value: float = -1e4):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.mask_value = float(mask_value)

        self.rope_dim = (self.head_dim // 2) * 2
        self.rope = RotaryEmbedding(self.rope_dim)

        self.ln1 = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.o_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)

        hidden = int(ffn_mult * self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.d_model),
        )

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, D = x.shape
        x_norm = self.ln1(x)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len=T, device=x.device, dtype=q.dtype)
        q_rope, q_pass = q[..., : self.rope_dim], q[..., self.rope_dim :]
        k_rope, k_pass = k[..., : self.rope_dim], k[..., self.rope_dim :]
        q = torch.cat([_apply_rope_1d(q_rope, cos, sin), q_pass], dim=-1)
        k = torch.cat([_apply_rope_1d(k_rope, cos, sin), k_pass], dim=-1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :].to(torch.bool), self.mask_value)
        attn = self.attn_drop(torch.softmax(attn_scores, dim=-1))

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.o_proj(out))
        x = x + self.drop(self.ffn(self.ln2(x)))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


# ----------------------------------------------------------------------------
# Peak <-> gene ISAB layer
# ----------------------------------------------------------------------------
class CrossAttentionWithBias(nn.Module):
    """Multi-head cross-attention with an additive bias and optional key padding mask."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, mask_value: float = -1e4):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.mask_value = float(mask_value)

    def forward(
        self,
        x_q: Tensor,  # [B, Nq, D]
        x_kv: Tensor,  # [B, Nk, D]
        attn_bias: Tensor,  # [B, H, Nq, Nk]
        key_padding_mask: Optional[Tensor] = None,  # [B, Nk], True = padding
    ) -> Tensor:
        B, Nq, D = x_q.shape
        Nk = x_kv.shape[1]
        q = self.q_proj(x_q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attn_bias.to(q.dtype)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], self.mask_value)
        attn = self.attn_drop(torch.softmax(attn_scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Nq, D)
        return self.o_proj(out)


class PeakGeneLayer(nn.Module):
    """One ISAB-style block: genes attend to peaks, then peaks attend to the updated genes (pre-LN)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ffn_mult: int = 2):
        super().__init__()
        self.cross_gene_to_peak = CrossAttentionWithBias(d_model, num_heads, dropout=dropout)
        self.cross_peak_to_gene = CrossAttentionWithBias(d_model, num_heads, dropout=dropout)

        hidden = int(ffn_mult * d_model)
        self.ffn_gene = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.ffn_peak = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))

        self.ln_gene_1 = nn.LayerNorm(d_model)
        self.ln_peak_1 = nn.LayerNorm(d_model)
        self.ln_gene_2 = nn.LayerNorm(d_model)
        self.ln_peak_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        peak_repr: Tensor,  # [B, P, D]
        gene_repr: Tensor,  # [B, G, D]
        peak_padding_mask: Tensor,  # [B, P]
        dist_bias: Tensor,  # [B, H, P, G]
    ) -> Tuple[Tensor, Tensor]:
        gene_delta = self.cross_gene_to_peak(
            self.ln_gene_1(gene_repr), self.ln_peak_1(peak_repr),
            attn_bias=dist_bias.permute(0, 1, 3, 2).contiguous(), key_padding_mask=peak_padding_mask,
        )
        gene_repr = gene_repr + self.dropout(gene_delta)

        peak_delta = self.cross_peak_to_gene(self.ln_peak_1(peak_repr), self.ln_gene_1(gene_repr), attn_bias=dist_bias)
        peak_repr = peak_repr + self.dropout(peak_delta)

        gene_ffn_out = self.ffn_gene(self.ln_gene_2(gene_repr))
        peak_ffn_out = self.ffn_peak(self.ln_peak_2(peak_repr))
        gene_repr = gene_repr + self.dropout(gene_ffn_out)
        peak_repr = peak_repr + self.dropout(peak_ffn_out)

        return peak_repr.masked_fill(peak_padding_mask.unsqueeze(-1), 0.0), gene_repr


# ----------------------------------------------------------------------------
# Gene-wise Poisson count head
# ----------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class PoissonCountHead(nn.Module):
    """Per-gene MLP + linear readout; softplus gives the Poisson rate (expected raw count)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = Mlp(d_model, d_model)
        # historical name kept for checkpoint compatibility
        self.softmax_output = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:  # [B, G, D] -> [B, G]
        return F.softplus(self.softmax_output(self.mlp(x)).squeeze(-1))
