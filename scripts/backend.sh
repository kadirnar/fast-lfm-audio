#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

backend="${1:-}"
case "$backend" in
    vllm) environment=vendor/envs/vllm; python_version=3.13; package=vllm==0.29.0 ;;
    sglang) environment=vendor/envs/sglang-py312; python_version=3.12; package=sglang==0.5.19 ;;
    *) printf 'Usage: bash scripts/backend.sh {vllm|sglang} [--setup|inference options]\n' >&2; exit 2 ;;
esac
shift

if [[ "${1:-}" == --setup ]]; then
    if [[ ! -d vendor/transformers/.git || ! -d vendor/fast-mimi/.git ]]; then
        printf 'Run bash scripts/setup.sh first to fetch the pinned dependencies.\n' >&2
        exit 1
    fi
    if [[ ! -x "$environment/bin/python" ]]; then
        uv venv --python "$python_version" "$environment"
    fi
    uv pip install --python "$environment/bin/python" "$package"
    # Override SGLang's Transformers/tokenizers pins with the newer audio PR and its tokenizer dependency.
    uv pip install --python "$environment/bin/python" -e vendor/transformers -e vendor/fast-mimi \
        accelerate==1.15.0 librosa==1.0.0 soundfile==0.13.1
    uv pip install --python "$environment/bin/python" --no-deps -e .
    if [[ ! -x vendor/envs/cuda/bin/python ]]; then
        uv venv --python 3.12 vendor/envs/cuda
    fi
    # Keep compiler and headers together, separate from Torch's CUDA runtime packages.
    uv pip install --python vendor/envs/cuda/bin/python \
        nvidia-cuda-nvcc==13.4.59 nvidia-cuda-crt==13.4.59 nvidia-nvvm==13.4.59 \
        nvidia-cuda-runtime==13.4.49 nvidia-cuda-cccl==13.3.4.2.1
    toolkit="$PWD/vendor/envs/cuda/lib/python3.12/site-packages/nvidia/cu13"
    [[ -e "$toolkit/lib64" ]] || ln -s lib "$toolkit/lib64"
    [[ -e "$toolkit/lib/libcudart.so" ]] || ln -s libcudart.so.13 "$toolkit/lib/libcudart.so"
    printf '%s ready.\n' "$backend"
    exit 0
fi

if [[ ! -x "$environment/bin/python" ]]; then
    printf 'Run: bash scripts/backend.sh %s --setup\n' "$backend" >&2
    exit 1
fi
export PATH="$PWD/$environment/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-$PWD/vendor/envs/cuda/lib/python3.12/site-packages/nvidia/cu13}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export FAST_LFM_NATIVE_BACKEND="$backend"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec "$environment/bin/python" -m fast_lfm_audio.cli --backend "$backend" --dtype bf16 "$@"
