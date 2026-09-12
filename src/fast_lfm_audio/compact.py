"""Lossless exponent packing for large reference-ordered FP32 GEMVs.

Mantissa and sign use one byte. Six five-bit exponent offsets share a uint32.
All bits are reconstructed before the original FP32 fused multiply/add sequence.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _restore(low, exponent, base: tl.constexpr):
    bits = ((low.to(tl.uint32) & 127) << 16) | ((low.to(tl.uint32) & 128) << 24) | ((exponent + base) << 23)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _chunked(X, W, H, Y, BASE: tl.constexpr, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, 16)
    total = tl.full((ROWS, 16), 0, tl.float32)
    for chunk in range(K // 512):
        accumulator = tl.full((ROWS, 16), 0, tl.float32)
        for group in tl.static_range(6):
            hidx = (
                tl.program_id(0) * ROWS * 16 * 24
                + (chunk * 6 + group) * ROWS * 16
                + tl.arange(0, ROWS)[:, None] * 16
                + lane[None, :]
            )
            word = tl.load(H + hidx, row[:, None] < N, other=0)
            for index in tl.static_range(6):
                if group * 6 + index < 32:
                    column = chunk * 512 + (group * 6 + index) * 16 + lane
                    offset = (
                        tl.program_id(0) * ROWS * K
                        + (chunk * 32 + group * 6 + index) * ROWS * 16
                        + tl.arange(0, ROWS)[:, None] * 16
                        + lane[None, :]
                    )
                    low = tl.load(W + offset, row[:, None] < N, other=0)
                    weight = _restore(low, (word >> (5 * index)) & 31, BASE)
                    value = tl.load(X + column)
                    accumulator = tl.fma(weight, value[None, :], accumulator)
        total = total + accumulator
    result = tl.gather(total, tl.full((ROWS, 1), 0, tl.int32), 1).reshape(ROWS)
    for lane_index in tl.static_range(1, 16):
        selected = tl.gather(total, tl.full((ROWS, 1), lane_index, tl.int32), 1).reshape(ROWS)
        result = result + selected
    tl.store(Y + row, result, row < N)


@triton.jit
def _tree(
    X, W, H, Y, BASE: tl.constexpr, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, LANES: tl.constexpr
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, LANES)
    accumulator = tl.full((ROWS, LANES), 0, tl.float32)
    for group in range(tl.cdiv(K, LANES * 6)):
        hidx = (
            tl.program_id(0) * tl.cdiv(K, LANES * 6) * ROWS * LANES
            + group * ROWS * LANES
            + tl.arange(0, ROWS)[:, None] * LANES
            + lane[None, :]
        )
        word = tl.load(H + hidx, (row[:, None] < N) & (group < tl.cdiv(K, LANES * 6)), other=0)
        for index in tl.static_range(6):
            column = (group * 6 + index) * LANES + lane
            offset = (
                tl.program_id(0) * ROWS * K
                + (group * 6 + index) * ROWS * LANES
                + tl.arange(0, ROWS)[:, None] * LANES
                + lane[None, :]
            )
            low = tl.load(W + offset, (row[:, None] < N) & (column[None, :] < K), other=0)
            weight = _restore(low, (word >> (5 * index)) & 31, BASE)
            value = tl.load(X + column, column < K, other=0)
            update = tl.fma(weight, value[None, :], accumulator)
            accumulator = tl.where(column[None, :] < K, update, accumulator)
    for shift in tl.static_range(LANES.bit_length() - 1):
        offset = LANES >> (shift + 1)
        indices = tl.broadcast_to(((lane + offset) % LANES)[None, :], (ROWS, LANES))
        accumulator = accumulator + tl.gather(accumulator, indices, 1)
    result = tl.gather(accumulator, tl.full((ROWS, 1), 0, tl.int32), 1).reshape(ROWS)
    # cuBLAS's alpha=1, beta=0 epilogue adds positive zero. Keep that operation
    # explicitly: negative products that underflow can otherwise leave -0 here.
    result = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, 0f00000000;",
        constraints="=f,f",
        args=[result],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(Y + row, result, row < N)


@triton.jit
def _low_and_range(W, L, Range, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    upper = tl.load(W + index, index < COUNT, other=0).to(tl.uint32)
    low = (upper & 127) | ((upper >> 8) & 128)
    exponent = (upper >> 7) & 255
    tl.store(L + index, low, index < COUNT)
    tl.store(Range + 2 * tl.program_id(0), tl.min(tl.where(index < COUNT, exponent, 255), 0))
    tl.store(Range + 2 * tl.program_id(0) + 1, tl.max(exponent, 0))


@triton.jit
def _encode_high(
    W,
    H,
    BASE: tl.constexpr,
    WORDS: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    LANES: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    word = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inner = word % (ROWS * LANES)
    iterations: tl.constexpr = K // LANES // CHUNKS
    groups: tl.constexpr = triton.cdiv(iterations, 6)
    group = word // (ROWS * LANES) % groups
    chunk = word // (ROWS * LANES * groups) % CHUNKS
    block = word // (ROWS * LANES * groups * CHUNKS)
    packed = tl.full((BLOCK,), 0, tl.uint32)
    for index in tl.static_range(6):
        step = group * 6 + index
        offset = block * ROWS * K + (chunk * iterations + step) * ROWS * LANES + inner
        upper = tl.load(W + offset, (word < WORDS) & (step < iterations), other=0).to(tl.uint32)
        # Keep each field explicitly five bits wide; unused tail fields are
        # ignored by GEMV and must not spill into neighboring valid fields.
        exponent = ((upper >> 7) - BASE) & 31
        packed = packed | (exponent << (index * 5))
    tl.store(H + word, packed, word < WORDS)


def pack(storage, *, k, rows, lanes):
    count = storage.numel()
    low = torch.empty(count, dtype=torch.uint8, device=storage.device)
    ranges = torch.empty((triton.cdiv(count, 4096), 2), dtype=torch.int32, device=storage.device)
    _low_and_range[(ranges.shape[0],)](storage, low, ranges, count, 4096)
    minimum, maximum = torch.stack((ranges[:, 0].min(), ranges[:, 1].max())).cpu().tolist()
    if maximum - minimum > 31:
        return None
    chunks = 4 if lanes is None else 1
    lanes = lanes or 16
    iterations = k // lanes // chunks
    words = (count // (rows * k)) * chunks * triton.cdiv(iterations, 6) * rows * lanes
    high = torch.empty(words, dtype=torch.uint32, device=storage.device)
    _encode_high[(triton.cdiv(words, 256),)](storage, high, minimum, words, k, rows, lanes, chunks, 256)
    return low, high, minimum


def gemv(value, output, compact, *, n, k, rows, lanes):
    low, high, base = compact
    kernel = _chunked if lanes is None else _tree
    options = (n, k, rows) if lanes is None else (n, k, rows, lanes)
    kernel[(triton.cdiv(n, rows),)](
        value, low, high, output, base, *options, num_warps=4, enable_fp_fusion=False
    )
    return output
