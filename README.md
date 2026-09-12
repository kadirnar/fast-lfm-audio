# Fast LFM Audio

Triton and CUDA graph inference for LFM2.5 Audio: text to speech, voice chat, and
transcription. FP32 by default, with FP16/BF16 and audio streaming support.

## Time to first audio (TTFA)

| Our backend / model precision | [Liquid Audio (original)](https://github.com/Liquid4All/liquid-audio/tree/19e65845923a7f136442c95137884ec61eb386aa) | **Fast LFM Audio (ours)** | **TTFA speedup** |
| --- | ---: | ---: | ---: |
| Transformers / FP32 | 122.62 ms | **43.92 ms** | **2.79x** |
| Transformers / FP16 | 116.51 ms | **36.48 ms** | **3.19x** |
| Transformers / BF16 | 115.71 ms | **36.62 ms** | **3.16x** |
| vLLM / BF16 | 115.71 ms | **48.02 ms** | **2.41x** |
| SGLang / BF16 | 115.71 ms | **50.84 ms** | **2.28x** |

Median TTFA on RTX 5070 Ti, batch one: same checkpoint, 4.904-second input,
64-token budget, 30 warm requests per mode. **Both repos stream 80 ms audio chunks.**
Each row compares with the original repo at the same model precision.
TTFA measures request start to the first CPU audio chunk, including preprocessing
and GPU-to-CPU transfer. Recording, network, model startup, and speaker latency
are excluded. First use requires warmup.

Precision refers to the model. Original Mimi uses FP32 with TF32 disabled; our
FP16/BF16 modes use a mixed BF16/FP32 decoder. Outputs can differ between repos,
backends, and precisions.
**10 ms TTFA has not been reached.**

## Setup

Requires Linux, an NVIDIA CUDA GPU, `git`, and `uv`.

```bash
git clone https://github.com/kadirnar/fast-lfm-audio.git
cd fast-lfm-audio
bash scripts/setup.sh
source .venv/bin/activate
```

Model weights download on first use.

## Run

```bash
# Text to speech
fast-lfm-audio --text 'Hello, this is a test.' --output speech.wav

# Voice chat
fast-lfm-audio --task chat --audio question.wav --output answer.wav

# Transcription
fast-lfm-audio --task asr --audio question.wav
```

Use `--dtype fp16` or `--dtype bf16` for reduced precision.
More options: `fast-lfm-audio --help`.

## Streaming

Use the Python API to receive audio during generation:

```python
from queue import SimpleQueue
from fast_lfm_audio import Pipeline

audio_queue = SimpleQueue()  # A separate playback worker consumes these chunks.
with Pipeline(dtype="fp32") as pipeline:
    text, waveform, output = pipeline(
        task="chat",
        audio="question.wav",
        on_audio=audio_queue.put,
        chunk_frames=1,
    )
audio_queue.put(None)  # Signal completion to the playback worker.
```

Chunks are CPU float32 tensors, `(1, samples)`, mono at 24 kHz. One frame contains
80 ms of audio; use `chunk_frames=4` for 320 ms chunks. Enqueue chunks promptly so
playback does not block generation.
Reuse the pipeline and warm representative requests before live use.

FP32 streaming keeps the model and Mimi in FP32. It preserves text/audio codes
on the checked inputs, with small waveform rounding differences. FP16/BF16 outputs
differ from FP32. All three backends support streaming with the fast-mimi codec.

## Optional backends

Experimental BF16 inference and streaming are available with vLLM and SGLang:

```bash
bash scripts/backend.sh vllm --setup
bash scripts/backend.sh vllm --text 'Hello from LFM.' --output speech.wav

bash scripts/backend.sh sglang --setup
bash scripts/backend.sh sglang --text 'Hello from LFM.' --output speech.wav
```

For streaming, run the Python example in the corresponding backend environment
and use `Pipeline(backend="vllm", dtype="bf16")` or
`Pipeline(backend="sglang", dtype="bf16")`. Native adapters require BF16; use the
default Transformers backend for FP32.

[Third-party notices](THIRD_PARTY_NOTICES.md)
