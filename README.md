# Fast LFM Audio

Triton and CUDA graph inference for LFM2.5 Audio: text to speech, voice chat, and
transcription. FP32 by default, with FP16/BF16 and audio streaming support.

## Latency by task

Audio tasks measure **time to first audio (TTFA)**. ASR measures **complete
transcription readiness**, since it produces no audio. All values are warm medians.

| Task | [Liquid Audio (original), FP32](https://github.com/Liquid4All/liquid-audio/tree/19e65845923a7f136442c95137884ec61eb386aa) | **Fast LFM Audio (ours), FP32** | **Speedup** |
| --- | ---: | ---: | ---: |
| TTS: UK female | 64.48 ms | **16.60 ms** | **3.88x** |
| TTS: UK male | 64.57 ms | **16.57 ms** | **3.90x** |
| TTS: US female | 64.48 ms | **16.55 ms** | **3.90x** |
| TTS: US male | 64.45 ms | **16.53 ms** | **3.90x** |
| Voice chat: audio input | 122.58 ms | **44.09 ms** | **2.78x** |
| Spoken chat: text input | 107.51 ms | **33.50 ms** | **3.21x** |
| ASR: transcript ready | 171.16 ms | **74.23 ms** | **2.31x** |

<details>
<summary>FP16/BF16 and native backends (milliseconds)</summary>

| Task | Original FP16 | Original BF16 | **Ours FP16** | **Ours BF16** | Ours vLLM BF16 | Ours SGLang BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TTS: UK female | 62.17 | 62.88 | **11.97** | **11.99** | 18.16 | 23.78 |
| TTS: UK male | 62.14 | 62.89 | **12.02** | **12.02** | 18.25 | 23.84 |
| TTS: US female | 62.24 | 62.85 | **11.97** | **12.01** | 18.20 | 23.75 |
| TTS: US male | 62.18 | 62.85 | **12.00** | **12.00** | 18.15 | 23.80 |
| Voice chat: audio input | 116.17 | 115.83 | **36.32** | **36.37** | 47.92 | 50.84 |
| Spoken chat: text input | 103.95 | 104.38 | **31.03** | **30.89** | 40.23 | 42.85 |
| ASR: transcript ready | 161.65 | 160.66 | **70.98** | **70.99** | 88.17 | 81.78 |

</details>

RTX 5070 Ti, batch one, same checkpoint, 64-token budget: 30 timed requests after
two warmups per task. Audio input is 4.904 seconds. Our FP32 backend is Transformers.
**Both repos stream 80 ms audio chunks.** TTFA measures request start to the first
CPU audio chunk, including preprocessing, decoding, and GPU-to-CPU transfer.
Recording/VAD, network, model loading, and speaker latency are excluded.

These measurements repeat the same inputs. The first FP32 voice-chat request after
TTS took **307.83 ms**, before warming to **44.09 ms**. New input shapes can require
further warmup.

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
