# Fast LFM Audio

Triton and CUDA graph inference for LFM2.5 Audio: text to speech, voice chat, and
transcription. FP32 by default, with FP16/BF16 and audio streaming support.

## Time to first audio (TTFA)

| Measurement | FP32 full decode | FP16 streaming | BF16 streaming | FP32 streaming |
| --- | ---: | ---: | ---: | ---: |
| TTFA median | 326.03 ms | 75.10 ms | 74.68 ms | **62.45 ms** |
| TTFA p95 | 327.29 ms | 75.89 ms | 75.25 ms | **63.20 ms** |
| TTFA speedup | 1x | 4.34x | 4.37x | **5.22x** |

RTX 5070 Ti, batch one, 4.904-second input, 64-token budget, 30 warm requests.
TTFA measures request start to the first CPU audio chunk, including preprocessing
and GPU-to-CPU transfer. Recording, network, model startup, and speaker latency
are excluded. First use requires warmup.

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
        chunk_frames=4,
    )
audio_queue.put(None)  # Signal completion to the playback worker.
```

Chunks are CPU float32 tensors, `(1, samples)`, mono at 24 kHz. Four frames contain
320 ms of audio. Enqueue chunks promptly so playback does not block generation.
Reuse the pipeline and warm representative requests before live use.

FP32 streaming keeps the model and Mimi in FP32. It preserves text/audio codes
on the checked inputs, with small waveform rounding differences (maximum measured
`6.26e-7`). FP16/BF16 outputs differ from FP32. Streaming uses the default
Transformers backend and fast-mimi codec.

## Optional backends

Experimental BF16 inference is also available through vLLM and SGLang:

```bash
bash scripts/backend.sh vllm --setup
bash scripts/backend.sh vllm --text 'Hello from LFM.' --output speech.wav

bash scripts/backend.sh sglang --setup
bash scripts/backend.sh sglang --text 'Hello from LFM.' --output speech.wav
```

[Third-party notices](THIRD_PARTY_NOTICES.md)
