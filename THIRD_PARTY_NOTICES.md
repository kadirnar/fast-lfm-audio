# Third-Party Notices

The text/audio scheduling in `src/fast_lfm_audio/backends/audio.py` and
`src/fast_lfm_audio/strict_generation.py`, and the
Conformer/Mimi block forwards in `src/fast_lfm_audio/strict.py` are adapted
from Hugging Face Transformers' LFM2-Audio implementation, PR #48249 at commit
`843101f38c800b98d49b704215f0b76b92e48e64`.

Copyright 2026 The HuggingFace Inc. team. All rights reserved.
Licensed under the [Apache License, Version 2.0](LICENSES/transformers-Apache-2.0.txt).
The adapted implementation separates the native backbone, audio sampler, and
request state. Dependencies and model weights retain their respective licenses.

The reference reduction ordering in `src/fast_lfm_audio/kernels.py` and softmax
ordering in `src/fast_lfm_audio/strict_attention.py` are adapted from the ATen CUDA
implementations in PyTorch 2.13.0 (`Reduce.cuh` and `PersistentSoftmax.cuh`).
The Triton implementations retain explicit FP32 rounding boundaries.
PyTorch is distributed under its [BSD-style license](LICENSES/pytorch-BSD-3-Clause.txt);
that file preserves the upstream copyright notices and license conditions.
