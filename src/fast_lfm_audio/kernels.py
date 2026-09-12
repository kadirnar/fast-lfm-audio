"""FP32 pointwise Triton kernels with the reference arithmetic order.

Known RMSNorm reductions reproduce ATen's accumulation and reduction order.
Calls recorded by autograd use the original expressions, including their backward
graphs. No weights, activation dtypes, or global precision flags are changed here.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rms_exact(X, W, Y, D: tl.constexpr, BW: tl.constexpr, EPS: tl.constexpr, WEIGHT_FIRST: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BW)
    a0 = tl.full((BW,), 0, tl.float32)
    a1 = tl.full((BW,), 0, tl.float32)
    a2 = tl.full((BW,), 0, tl.float32)
    a3 = tl.full((BW,), 0, tl.float32)
    if D >= 128:
        for block in range(D // (BW * 4)):
            index = row * D + block * BW * 4 + lane * 4
            v0 = tl.load(X + index)
            v1 = tl.load(X + index + 1)
            v2 = tl.load(X + index + 2)
            v3 = tl.load(X + index + 3)
            a0 = a0 + v0 * v0
            a1 = a1 + v1 * v1
            a2 = a2 + v2 * v2
            a3 = a3 + v3 * v3
    else:
        v0 = tl.load(X + row * D + lane)
        a0 = a0 + v0 * v0
        if D > BW:
            v1 = tl.load(X + row * D + lane + BW)
            a1 = a1 + v1 * v1
    value = ((a0 + a1) + a2) + a3
    # Explicit decreasing offsets prevent layout-dependent reassociation by tl.sum.
    for step in tl.static_range(BW.bit_length() - 1):
        other = tl.gather(value, (lane + (BW >> (step + 1))) % BW, 0)
        value = value + other
    variance = tl.sum(tl.gather(value, tl.full((1,), 0, tl.int32), 0), 0) * (1.0 / D)
    scale = libdevice.rsqrt(variance + EPS)
    col = tl.arange(0, D)
    x = tl.load(X + row * D + col)
    weight = tl.load(W + col)
    scaled = x * scale
    result = weight * scaled if WEIGHT_FIRST else scaled * weight
    tl.store(Y + row * D + col, result)


@triton.jit
def _rms_scale(
    X,
    V,
    W,
    Y,
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index, index < N, other=0)
    variance = tl.load(V + index // D, index < N, other=0)
    weight = tl.load(W + index % D, index < N, other=0)
    scaled = value * libdevice.rsqrt(variance + EPS)
    if WEIGHT_FIRST:
        result = weight * scaled
    else:
        result = scaled * weight
    tl.store(Y + index, result, index < N)


@triton.jit
def _silu_mul(X, G, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index, index < N, other=0)
    gate = tl.load(G + index, index < N, other=0)
    activated = tl.div_rn(value, 1.0 + libdevice.exp(-value))
    tl.store(Y + index, activated * gate, index < N)


@triton.jit
def _scaled_add(R, X, W, Y, N: tl.constexpr, D: tl.constexpr, SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    residual = tl.load(R + index, index < N, other=0)
    value = tl.load(X + index, index < N, other=0)
    if SCALAR:
        scale = 0.5
    else:
        scale = tl.load(W + index % D, index < N, other=0)
    scaled = scale * value
    tl.store(Y + index, residual + scaled, index < N)


def _eligible(*values):
    return (
        not torch.is_grad_enabled()
        and not torch.is_autocast_enabled("cuda")
        and all(value.is_cuda and value.dtype == torch.float32 and value.is_contiguous() for value in values)
        and all(value.device == values[0].device for value in values)
    )


def rms_norm(value, weight, eps, *, weight_first=False):
    """Compute FP32 RMSNorm, preserving ATen's square and mean arithmetic."""
    # Triton's reciprocal-square-root lowering flushes subnormal arguments.
    # A normal positive epsilon keeps every finite variance on its exact path.
    if (
        not _eligible(value, weight)
        or weight.ndim != 1
        or value.shape[-1] != weight.numel()
        or not torch.finfo(torch.float32).tiny <= eps <= torch.finfo(torch.float32).max
    ):
        scaled = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        return weight * scaled if weight_first else scaled * weight
    dim = value.shape[-1]
    if (
        value.numel()
        and dim in (32, 64, 128, 256, 512, 1024, 2048)
        and torch.__version__ == "2.13.0+cu130"
        and triton.__version__ == "3.7.1"
    ):
        # ATen Reduce.cuh: four independent vector accumulators, then the
        # decreasing-offset block/warp sum. Block width depends on row count.
        rows = value.numel() // dim
        dim0 = dim // 4 if dim >= 128 else dim
        width = min(dim0, 32)
        height = min(1 << (rows.bit_length() - 1), 512 // width)
        width = min(dim0, 512 // height)
        output = torch.empty_like(value)
        with torch.cuda.device(value.device):
            _rms_exact[(rows,)](
                value, weight, output, dim, width, eps, weight_first, enable_fp_fusion=False, num_warps=4
            )
        return output
    variance = value.square().mean(-1, keepdim=True)
    output = torch.empty_like(value)
    if value.numel():
        with torch.cuda.device(value.device):
            _rms_scale[(triton.cdiv(value.numel(), 256),)](
                value,
                variance,
                weight,
                output,
                value.numel(),
                value.shape[-1],
                eps,
                weight_first,
                256,
                enable_fp_fusion=False,
            )
    return output


def silu_mul(value, gate):
    """Fuse SiLU and its gate multiplication without approximate exponentials."""
    if not _eligible(value, gate) or value.shape != gate.shape:
        return F.silu(value) * gate
    output = torch.empty_like(value)
    if value.numel():
        with torch.cuda.device(value.device):
            _silu_mul[(triton.cdiv(value.numel(), 256),)](
                value,
                gate,
                output,
                value.numel(),
                256,
                enable_fp_fusion=False,
            )
    return output


def scaled_add(residual, value, scale=0.5):
    """Fuse residual + scale * value with two separately rounded FP32 operations."""
    scalar = isinstance(scale, (int, float))
    values = (residual, value) if scalar else (residual, value, scale)
    if (
        not _eligible(*values)
        or residual.shape != value.shape
        or (scalar and scale != 0.5)
        or (not scalar and (scale.ndim != 1 or scale.numel() != value.shape[-1]))
    ):
        return residual + scale * value
    output = torch.empty_like(value)
    if value.numel():
        with torch.cuda.device(value.device):
            _scaled_add[(triton.cdiv(value.numel(), 256),)](
                residual,
                value,
                value if scalar else scale,
                output,
                value.numel(),
                value.shape[-1],
                scalar,
                256,
                enable_fp_fusion=False,
            )
    return output
