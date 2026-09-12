"""Lossless weight storage and reference-ordered FP32 matrix-vector products.

Only the measured cuBLAS dispatch shapes on the pinned RTX 5070 Ti runtime are
enabled. All arithmetic is FP32; uint16 storage drops only verified zero bits.
The original FP32 Parameters remain available to autograd and unsupported calls.
"""

import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .kernels import _eligible


def supported_runtime(device):
    if torch.__version__ != "2.13.0+cu130" or triton.__version__ != "3.7.1":
        return False
    try:
        package = distribution("nvidia-cublas")
        if package.version != "13.1.1.3":
            return False
        names = {"libcublas.so.13", "libcublasLt.so.13"}
        expected = {
            str(Path(package.locate_file(path)).resolve())
            for path in package.files or ()
            if path.name in names
        }
        loaded = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(name in line for name in names)
        }
        # A matching wheel version cannot certify an LD_LIBRARY_PATH/PRELOAD
        # override. Require both mapped cuBLAS libraries from that wheel.
        if len(expected) != 2 or loaded != expected:
            return False
    except (PackageNotFoundError, OSError):
        return False
    properties = torch.cuda.get_device_properties(device)
    return (
        properties.name == "NVIDIA GeForce RTX 5070 Ti"
        and (properties.major, properties.minor) == (12, 0)
        and properties.multi_processor_count == 70
    )


@triton.jit
def _gemv_chunked(X, W, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, 16)
    total = tl.full((ROWS, 16), 0, tl.float32)
    for chunk in range(K // 512):
        accumulator = tl.full((ROWS, 16), 0, tl.float32)
        for index in tl.static_range(32):
            column = chunk * 512 + index * 16 + lane
            offset = (
                tl.program_id(0) * ROWS * K
                + (chunk * 32 + index) * ROWS * 16
                + tl.arange(0, ROWS)[:, None] * 16
                + lane[None, :]
            )
            bits = tl.load(W + offset, row[:, None] < N, other=0).to(tl.uint32) << 16
            weight = bits.to(tl.float32, bitcast=True)
            value = tl.load(X + column)
            accumulator = tl.fma(weight, value[None, :], accumulator)
        total = total + accumulator
    result = tl.gather(total, tl.full((ROWS, 1), 0, tl.int32), 1).reshape(ROWS)
    for lane_index in tl.static_range(1, 16):
        selected = tl.gather(total, tl.full((ROWS, 1), lane_index, tl.int32), 1).reshape(ROWS)
        result = result + selected
    tl.store(Y + row, result, row < N)


@triton.jit
def _gemv_tree(X, W, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, LANES: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, LANES)
    accumulator = tl.full((ROWS, LANES), 0, tl.float32)
    for chunk in range(tl.cdiv(K, LANES * 16)):
        for index in tl.static_range(16):
            column = (chunk * 16 + index) * LANES + lane
            offset = (
                tl.program_id(0) * ROWS * K
                + (chunk * 16 + index) * ROWS * LANES
                + tl.arange(0, ROWS)[:, None] * LANES
                + lane[None, :]
            )
            bits = tl.load(W + offset, (row[:, None] < N) & (column[None, :] < K), other=0)
            weight = (bits.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
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


# cuBLAS's choice depends on shape and device. Do not extrapolate these choices
# to other shapes, GPUs or library versions without new bitwise validation.
_CHUNKED = {(2048, 2048), (6144, 2048), (8192, 2048), (65536, 2048)}
_TREE = {
    (2048, 8192): 32,
    (2816, 1024): 8,
    (1024, 2816): 32,
    (512, 2048): 32,
    (1536, 1024): 16,
    (1024, 1024): 32,
    (2049, 1024): 8,
}


class PackedLinear:
    def __init__(self, module, original=None):
        self.module = module
        self.original = original or module.forward
        self.weight = module.weight
        self.parameter_version = self.weight._version
        self.n, self.k = self.weight.shape
        self.lanes = _TREE.get((self.n, self.k))
        self.rows = 16 if self.lanes in (None, 8) else 4
        lanes = self.lanes or 16
        with torch.inference_mode():
            data = (self.weight.view(torch.int32) >> 16).to(torch.uint16)
            if self.n % self.rows:
                padded = torch.zeros(
                    (triton.cdiv(self.n, self.rows) * self.rows, self.k),
                    device=data.device,
                    dtype=torch.int16,
                ).view(torch.uint16)
                padded[: self.n].copy_(data)
                data = padded
            self.storage = (
                data.reshape(-1, self.rows, self.k // lanes, lanes).permute(0, 2, 1, 3).contiguous()
            )
            self.compact = None
            if (
                self.n * self.k >= 8192 * 2048
                and os.environ.get("FAST_LFM_STRICT_COMPACT_WEIGHTS", "1") == "1"
            ):
                from .compact import pack

                self.compact = pack(self.storage, k=self.k, rows=self.rows, lanes=self.lanes)
                if self.compact is not None:
                    self.storage = None

    @classmethod
    def create(cls, module, *, projection=False):
        if projection and type(module) is not torch.nn.Embedding:
            return None
        if not projection and (type(module) is not torch.nn.Linear or module.bias is not None):
            return None
        weight = module.weight
        if (
            not weight.is_cuda
            or weight.dtype != torch.float32
            or not weight.is_contiguous()
            or weight.is_inference()
        ):
            return None
        if tuple(weight.shape) not in _CHUNKED | _TREE.keys():
            return None
        with torch.no_grad():
            if not bool(((weight.view(torch.int32) & 0xFFFF) == 0).all()):
                return None
        original = (lambda value: F.linear(value, module.weight)) if projection else None
        return cls(module, original)

    def unchanged(self):
        return self.module.weight is self.weight and self.weight._version == self.parameter_version

    def __call__(self, value):
        if (
            self.module.training
            or not _eligible(value, self.weight)
            or value.numel() != self.k
            or value.shape[-1] != self.k
            or not self.unchanged()
        ):
            return self.original(value)
        output = torch.empty((*value.shape[:-1], self.n), dtype=torch.float32, device=value.device)
        with torch.cuda.device(value.device):
            if self.compact is not None:
                from .compact import gemv

                return gemv(value, output, self.compact, n=self.n, k=self.k, rows=self.rows, lanes=self.lanes)
            elif self.lanes is None:
                _gemv_chunked[(triton.cdiv(self.n, self.rows),)](
                    value,
                    self.storage,
                    output,
                    self.n,
                    self.k,
                    self.rows,
                    enable_fp_fusion=False,
                    num_warps=4,
                )
            else:
                _gemv_tree[(triton.cdiv(self.n, self.rows),)](
                    value,
                    self.storage,
                    output,
                    self.n,
                    self.k,
                    self.rows,
                    self.lanes,
                    enable_fp_fusion=False,
                    num_warps=4,
                )
        return output
