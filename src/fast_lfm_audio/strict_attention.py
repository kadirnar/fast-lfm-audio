"""Reference-ordered FP32 attention for the pinned eight-codebook Depthformer.

The original cuBLAS two-lane dot products, CUDA softmax reductions, exponential
and division rounding are retained. This module is selected only by strict.py
on the validated runtime; unsupported shapes and all autograd calls fall back.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _attention(
    Q,
    K,
    V,
    Y,
    L: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KH: tl.constexpr,
    KT: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
):
    head = tl.program_id(0) * BH + tl.arange(0, BH)
    pos = tl.arange(0, BN)
    a = tl.full((BH, BN), 0, tl.float32)
    b = tl.full((BH, BN), 0, tl.float32)
    for dim in tl.static_range(D // 2):
        q0 = tl.load(Q + head * QS + 2 * dim) * SCALE
        q1 = tl.load(Q + head * QS + 2 * dim + 1) * SCALE
        k0 = (
            tl.load(K + head[:, None] // 4 * KH + pos[None, :] * KT + 2 * dim, pos[None, :] < L, other=0)
            * SCALE
        )
        k1 = (
            tl.load(K + head[:, None] // 4 * KH + pos[None, :] * KT + 2 * dim + 1, pos[None, :] < L, other=0)
            * SCALE
        )
        a = tl.fma(k0, q0[:, None], a)
        b = tl.fma(k1, q1[:, None], b)
    scores = a + b
    scores = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, 0f00000000;",
        constraints="=f,f",
        args=[scores],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    scores = tl.where(pos[None, :] < L, scores, float("-inf"))
    maximum = scores
    for step in tl.static_range(BN.bit_length() - 1):
        offset = BN >> (step + 1)
        other = tl.gather(maximum, tl.broadcast_to((pos ^ offset)[None, :], (BH, BN)), 1)
        maximum = tl.where(maximum < other, other, maximum)
    numerator = libdevice.exp(scores - maximum)
    denominator = numerator
    for step in tl.static_range(BN.bit_length() - 1):
        offset = BN >> (step + 1)
        denominator = denominator + tl.gather(
            denominator, tl.broadcast_to((pos ^ offset)[None, :], (BH, BN)), 1
        )
    probabilities = tl.div_rn(numerator, denominator)
    all_negative_inf = tl.sum((scores != float("-inf")).to(tl.int32), 1) == 0
    probabilities = tl.where(all_negative_inf[:, None], 0.0, probabilities)
    dim = tl.arange(0, D)
    a = tl.full((BH, D), 0, tl.float32)
    b = tl.full((BH, D), 0, tl.float32)
    if L == 1:
        p = tl.gather(probabilities, tl.full((BH, 1), 0, tl.int32), 1)
        a = p * tl.load(V + head[:, None] // 4 * VH + dim[None, :])
    else:
        for position in tl.static_range(tl.cdiv(L, 2)):
            p0 = tl.gather(probabilities, tl.full((BH, 1), 2 * position, tl.int32), 1)
            v0 = tl.load(V + head[:, None] // 4 * VH + 2 * position * VT + dim[None, :])
            a = tl.fma(p0, v0, a)
            if 2 * position + 1 < L:
                p1 = tl.gather(probabilities, tl.full((BH, 1), 2 * position + 1, tl.int32), 1)
                v1 = tl.load(V + head[:, None] // 4 * VH + (2 * position + 1) * VT + dim[None, :])
                b = tl.fma(p1, v1, b)
        a = a + b
        a = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, 0f00000000;",
            constraints="=f,f",
            args=[a],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(Y + head[:, None] * D + dim[None, :], a)


def depth_attention(query, key, value):
    length = key.shape[-2]
    output = torch.empty_like(query)
    with torch.cuda.device(query.device):
        _attention[(8,)](
            query,
            key,
            value,
            output,
            length,
            32,
            query.stride(1),
            key.stride(1),
            key.stride(2),
            value.stride(1),
            value.stride(2),
            math.sqrt(1 / math.sqrt(32)),
            triton.next_power_of_2(length),
            4,
            enable_fp_fusion=False,
            num_warps=4,
        )
    return output
