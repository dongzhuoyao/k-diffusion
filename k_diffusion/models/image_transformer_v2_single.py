"""k-diffusion transformer diffusion models, version 2."""

from dataclasses import dataclass
from functools import lru_cache, reduce
import math
from typing import Union

from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F


try:
    import natten
except ImportError:
    natten = None

try:
    import flash_attn
except ImportError:
    flash_attn = None


### FLAGS
from contextlib import contextmanager
from functools import update_wrapper
import os
import threading

import torch


def get_use_compile():
    return os.environ.get("K_DIFFUSION_USE_COMPILE", "1") == "1"


def get_use_flash_attention_2():
    return os.environ.get("K_DIFFUSION_USE_FLASH_2", "1") == "1"


state = threading.local()
state.checkpointing = False


@contextmanager
def checkpointing(enable=True):
    try:
        old_checkpointing, state.checkpointing = state.checkpointing, enable
        yield
    finally:
        state.checkpointing = old_checkpointing


def get_checkpointing():
    return getattr(state, "checkpointing", False)


class compile_wrap:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        if get_use_compile():
            try:
                self._compiled_function = torch.compile(
                    self.function, *self.args, **self.kwargs
                )
            except RuntimeError:
                self._compiled_function = self.function
        else:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        return self.compiled_function(*args, **kwargs)


### FLAGS end


if get_use_compile():
    torch._dynamo.config.cache_size_limit = max(
        64, torch._dynamo.config.cache_size_limit
    )
    torch._dynamo.config.suppress_errors = True


# Helpers


### FLOPS

from contextlib import contextmanager
import math
import threading


state = threading.local()
state.flop_counter = None


@contextmanager
def flop_counter(enable=True):
    try:
        old_flop_counter = state.flop_counter
        state.flop_counter = FlopCounter() if enable else None
        yield state.flop_counter
    finally:
        state.flop_counter = old_flop_counter


class FlopCounter:
    def __init__(self):
        self.ops = []

    def op(self, op, *args, **kwargs):
        self.ops.append((op, args, kwargs))

    @property
    def flops(self):
        flops = 0
        for op, args, kwargs in self.ops:
            flops += op(*args, **kwargs)
        return flops


def op(op, *args, **kwargs):
    if getattr(state, "flop_counter", None):
        state.flop_counter.op(op, *args, **kwargs)


def op_linear(x, weight):
    return math.prod(x) * weight[0]


def op_attention(q, k, v):
    *b, s_q, d_q = q
    *b, s_k, d_k = k
    *b, s_v, d_v = v
    return math.prod(b) * s_q * s_k * (d_q + d_v)


def op_natten(q, k, v, kernel_size):
    *q_rest, d_q = q
    *_, d_v = v
    return math.prod(q_rest) * (d_q + d_v) * kernel_size**2


####FLOPS end


### RoPE
def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_axial_pos(
    h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None
):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


###RoPE end


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer(
            "weight", torch.randn([out_features // 2, in_features]) * std
        )

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def checkpoint(function, *args, **kwargs):
    if get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)


# Param tags


def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


# Kernels


@compile_wrap
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


@compile_wrap
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


@compile_wrap
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


# Layers


class Linear(nn.Linear):
    def forward(self, x):
        op(op_linear, x.shape, self.weight.shape)
        return super().forward(x)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        op(op_linear, x.shape, self.weight.shape)
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


# Rotary position embeddings


@compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


@compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


# Shifted window attention


def window(window_size, x):
    *b, h, w, c = x.shape
    x = torch.reshape(
        x,
        (*b, h // window_size, window_size, w // window_size, window_size, c),
    )
    x = torch.permute(
        x,
        (*range(len(b)), -5, -3, -4, -2, -1),
    )
    return x


def unwindow(x):
    *b, h, w, wh, ww, c = x.shape
    x = torch.permute(x, (*range(len(b)), -5, -3, -4, -2, -1))
    x = torch.reshape(x, (*b, h * wh, w * ww, c))
    return x


def shifted_window(window_size, window_shift, x):
    x = torch.roll(x, shifts=(window_shift, window_shift), dims=(-2, -3))
    windows = window(window_size, x)
    return windows


def shifted_unwindow(window_shift, x):
    x = unwindow(x)
    x = torch.roll(x, shifts=(-window_shift, -window_shift), dims=(-2, -3))
    return x


@lru_cache
def make_shifted_window_masks(n_h_w, n_w_w, w_h, w_w, shift, device=None):
    ph_coords = torch.arange(n_h_w, device=device)
    pw_coords = torch.arange(n_w_w, device=device)
    h_coords = torch.arange(w_h, device=device)
    w_coords = torch.arange(w_w, device=device)
    patch_h, patch_w, q_h, q_w, k_h, k_w = torch.meshgrid(
        ph_coords,
        pw_coords,
        h_coords,
        w_coords,
        h_coords,
        w_coords,
        indexing="ij",
    )
    is_top_patch = patch_h == 0
    is_left_patch = patch_w == 0
    q_above_shift = q_h < shift
    k_above_shift = k_h < shift
    q_left_of_shift = q_w < shift
    k_left_of_shift = k_w < shift
    m_corner = (
        is_left_patch
        & is_top_patch
        & (q_left_of_shift == k_left_of_shift)
        & (q_above_shift == k_above_shift)
    )
    m_left = is_left_patch & ~is_top_patch & (q_left_of_shift == k_left_of_shift)
    m_top = ~is_left_patch & is_top_patch & (q_above_shift == k_above_shift)
    m_rest = ~is_left_patch & ~is_top_patch
    m = m_corner | m_left | m_top | m_rest
    return m


def apply_window_attention(window_size, window_shift, q, k, v, scale=None):
    # prep windows and masks
    q_windows = shifted_window(window_size, window_shift, q)
    k_windows = shifted_window(window_size, window_shift, k)
    v_windows = shifted_window(window_size, window_shift, v)
    b, heads, h, w, wh, ww, d_head = q_windows.shape
    mask = make_shifted_window_masks(h, w, wh, ww, window_shift, device=q.device)
    q_seqs = torch.reshape(q_windows, (b, heads, h, w, wh * ww, d_head))
    k_seqs = torch.reshape(k_windows, (b, heads, h, w, wh * ww, d_head))
    v_seqs = torch.reshape(v_windows, (b, heads, h, w, wh * ww, d_head))
    mask = torch.reshape(mask, (h, w, wh * ww, wh * ww))

    # do the attention here
    op(op_attention, q_seqs.shape, k_seqs.shape, v_seqs.shape)
    qkv = F.scaled_dot_product_attention(q_seqs, k_seqs, v_seqs, mask, scale=scale)

    # unwindow
    qkv = torch.reshape(qkv, (b, heads, h, w, wh, ww, d_head))
    return shifted_unwindow(window_shift, qkv)


# Transformer layers


def use_flash_2(x):
    if not get_use_flash_attention_2():
        return False
    if flash_attn is None:
        return False
    if x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
        theta = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = apply_rotary_emb_(qkv, theta)
            flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            op(op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(
                x, "n (h w) nh e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2]
            )
        else:
            q, k, v = rearrange(
                qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head
            )
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta = theta.movedim(-2, -3)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            op(op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(
                x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2]
            )
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
        theta = self.pos_emb(pos).movedim(-2, -4)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        op(op_natten, q.shape, k.shape, v.shape, self.kernel_size)
        qk = natten.functional.natten2dqk(q, k, self.kernel_size, 1)
        a = torch.softmax(qk, dim=-1).to(v.dtype)
        x = natten.functional.natten2dav(a, v, self.kernel_size, 1)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class ShiftedWindowSelfAttentionBlock(nn.Module):
    def __init__(
        self, d_model, d_head, cond_features, window_size, window_shift, dropout=0.0
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.window_size = window_size
        self.window_shift = window_shift
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, window_size={self.window_size}, window_shift={self.window_shift}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
        theta = self.pos_emb(pos).movedim(-2, -4)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        x = apply_window_attention(
            self.window_size, self.window_shift, q, k, v, scale=1.0
        )
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(
            d_model, d_head, cond_features, dropout=dropout
        )
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(
            d_model, d_head, cond_features, kernel_size, dropout=dropout
        )
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class ShiftedWindowTransformerLayer(nn.Module):
    def __init__(
        self, d_model, d_ff, d_head, cond_features, window_size, index, dropout=0.0
    ):
        super().__init__()
        window_shift = window_size // 2 if index % 2 == 1 else 0
        self.self_attn = ShiftedWindowSelfAttentionBlock(
            d_model, d_head, cond_features, window_size, window_shift, dropout=dropout
        )
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NoAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.ff, x, cond)
        return x


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


# Mapping network


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList(
            [
                MappingFeedForwardBlock(d_model, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


# Token merging and splitting


class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(
            Linear(in_features * self.h * self.w, out_features, bias=False)
        )

    def forward(self, x):
        x = rearrange(
            x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w
        )
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(
            Linear(in_features, out_features * self.h * self.w, bias=False)
        )

    def forward(self, x):
        x = self.proj(x)
        return rearrange(
            x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w
        )


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(
            Linear(in_features, out_features * self.h * self.w, bias=False)
        )
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(
            x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w
        )
        return torch.lerp(skip, x, self.fac.to(x.dtype))


# Configuration


@dataclass
class GlobalAttentionSpec:
    d_head: int


@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class ShiftedWindowAttentionSpec:
    d_head: int
    window_size: int


@dataclass
class NoAttentionSpec:
    pass


@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[
        GlobalAttentionSpec,
        NeighborhoodAttentionSpec,
        ShiftedWindowAttentionSpec,
        NoAttentionSpec,
    ]
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float


# Model class


class ImageTransformerDenoiserModelV2(nn.Module):
    def __init__(
        self,
        levels,
        mapping,
        in_channels,
        out_channels,
        patch_size,
        num_classes=0,
        mapping_cond_dim=0,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.aug_emb = FourierFeatures(9, mapping.width)
        self.aug_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.class_emb = (
            nn.Embedding(num_classes, mapping.width) if num_classes else None
        )
        self.mapping_cond_in_proj = (
            Linear(mapping_cond_dim, mapping.width, bias=False)
            if mapping_cond_dim
            else None
        )
        self.mapping = tag_module(
            MappingNetwork(
                mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout
            ),
            "mapping",
        )

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(
                    spec.width,
                    spec.d_ff,
                    spec.self_attn.d_head,
                    mapping.width,
                    dropout=spec.dropout,
                )
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(
                    spec.width,
                    spec.d_ff,
                    spec.self_attn.d_head,
                    mapping.width,
                    spec.self_attn.kernel_size,
                    dropout=spec.dropout,
                )
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(
                    spec.width,
                    spec.d_ff,
                    spec.self_attn.d_head,
                    mapping.width,
                    spec.self_attn.window_size,
                    i,
                    dropout=spec.dropout,
                )
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(
                    spec.width, spec.d_ff, mapping.width, dropout=spec.dropout
                )
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(
                    Level([layer_factory(i) for i in range(spec.depth)])
                )
                self.up_levels.append(
                    Level([layer_factory(i + spec.depth) for i in range(spec.depth)])
                )
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList(
            [
                TokenMerge(spec_1.width, spec_2.width)
                for spec_1, spec_2 in zip(levels[:-1], levels[1:])
            ]
        )
        self.splits = nn.ModuleList(
            [
                TokenSplit(spec_2.width, spec_1.width)
                for spec_1, spec_2 in zip(levels[:-1], levels[1:])
            ]
        )

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(
            levels[0].width, out_channels, patch_size
        )
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(
            lambda tags: "wd" not in tags and "mapping" not in tags, self
        )
        mapping_wd = filter_params(
            lambda tags: "wd" in tags and "mapping" in tags, self
        )
        mapping_no_wd = filter_params(
            lambda tags: "wd" not in tags and "mapping" in tags, self
        )
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {
                "params": list(mapping_no_wd),
                "lr": base_lr * mapping_lr_scale,
                "weight_decay": 0.0,
            },
        ]
        return groups

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None):
        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        # TODO: pixel aspect ratio for nonsquare patches
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(
            x.shape[-3], x.shape[-2], 2
        )

        # Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")
        if mapping_cond is None and self.mapping_cond_in_proj is not None:
            raise ValueError("mapping_cond must be specified if mapping_cond_dim > 0")

        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        mapping_emb = (
            self.mapping_cond_in_proj(mapping_cond)
            if self.mapping_cond_in_proj is not None
            else 0
        )
        cond = self.mapping(time_emb + aug_emb + class_emb + mapping_emb)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(
            list(zip(self.up_levels, self.splits, skips, poses))
        ):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)

        return x


class HDiT(nn.Module):
    def __init__(
        self,
        ###
        input_channels=3,
        patch_size=[4, 4],
        num_classes=0,
        widths=[384, 768, 1536],
        depths=[2, 2, 16],
        dropout_rate=[0.0, 0.0, 0.0],
        mapping_dropout_rate=0.0,
        d_ffs=None,
        self_attns=None,
        mapping_depth=256,
        mapping_width=2,
        mapping_d_ff=None,
        mapping_cond_dim=0,
    ):
        super().__init__()
        if mapping_d_ff is None:
            mapping_d_ff = mapping_width * 3
        if d_ffs is None:
            d_ffs = []
            for width in widths:
                d_ffs.append(width * 3)

        if self_attns is None:
            self_attns = []
            if False:
                default_neighborhood = {
                    "type": "neighborhood",
                    "d_head": 64,
                    "kernel_size": 7,
                }
            else:
                default_neighborhood = {"type": "global", "d_head": 64}
            default_global = {"type": "global", "d_head": 64}
            for i in range(len(widths)):
                self_attns.append(
                    default_neighborhood if i < len(widths) - 1 else default_global
                )

        assert len(widths) == len(depths)
        assert len(widths) == len(d_ffs)
        assert len(widths) == len(self_attns)
        assert len(widths) == len(dropout_rate)
        levels = []

        for depth, width, d_ff, self_attn, dropout in zip(
            depths,
            widths,
            d_ffs,
            self_attns,
            dropout_rate,
        ):
            if self_attn["type"] == "global":
                self_attn = GlobalAttentionSpec(self_attn.get("d_head", 64))
            elif self_attn["type"] == "neighborhood":
                self_attn = NeighborhoodAttentionSpec(
                    self_attn.get("d_head", 64), self_attn.get("kernel_size", 7)
                )
            elif self_attn["type"] == "shifted-window":
                self_attn = ShiftedWindowAttentionSpec(
                    self_attn.get("d_head", 64), self_attn["window_size"]
                )
            elif self_attn["type"] == "none":
                self_attn = NoAttentionSpec()
            else:
                raise ValueError(f'unsupported self attention type {self_attn["type"]}')
            levels.append(LevelSpec(depth, width, d_ff, self_attn, dropout))
        mapping = MappingSpec(
            mapping_depth,
            mapping_width,
            mapping_d_ff,
            mapping_dropout_rate,
        )
        print("Mapping: ", mapping)
        model = ImageTransformerDenoiserModelV2(
            levels=levels,
            mapping=mapping,
            in_channels=input_channels,
            out_channels=input_channels,
            patch_size=patch_size,
            num_classes=num_classes + 1 if num_classes else 0,
            mapping_cond_dim=mapping_cond_dim,
        )
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    model = HDiT()
    x = torch.randn(1, 3, 32, 32)
    sigma = torch.randn(1)
    y = model(x, sigma)
    print(y.shape)
