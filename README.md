# Fast LFM Audio

LFM2.5 Audio 1.5B inference for text to speech, voice chat, and transcription.
Handwritten Triton kernels and CUDA graphs accelerate the default FP32 path.
FP32, FP16, and BF16 support audio streaming through the Python API.

## Setup

Requires Linux, an NVIDIA CUDA GPU, `git`, and `uv`.

```bash
git clone https://github.com/kadirnar/fast-lfm-audio.git
cd fast-lfm-audio
bash scripts/setup.sh
source .venv/bin/activate
```

Setup installs the pinned inference dependencies in `.venv`, including
[Transformers PR #48249](https://github.com/huggingface/transformers/pull/48249)
and [fast-mimi](https://github.com/kadirnar/fast-mimi).
The model and Mimi checkpoints download on first use.

## Run

```bash
# Text to speech
fast-lfm-audio --text 'Hello, this is a test.' --output speech.wav

# Spoken answer to audio
fast-lfm-audio --task chat --audio question.wav --output answer.wav

# Transcription
fast-lfm-audio --task asr --audio question.wav
```

The default is `--dtype fp32`. Select `--dtype fp16` or `--dtype bf16` explicitly
for reduced precision. FP32 output uses floating-point WAV files.
Chat also accepts `--text`. TTS voices: `UK female` (default), `UK male`,
`US female`, and `US male`; select with `--voice`.
`--max-new-tokens` defaults to 512 and can truncate a response.
More options: `fast-lfm-audio --help`.

## Python

```python
import soundfile as sf
from fast_lfm_audio import Pipeline

with Pipeline(dtype="fp32") as pipeline:
    text, waveform, output = pipeline(text="Hello from exact FP32 inference.")
    sf.write("speech.wav", waveform[0].cpu().numpy(), 24000, subtype="FLOAT")
    text, waveform, output = pipeline(task="chat", audio="question.wav")
```

Reuse the same instance for subsequent requests. Audio is mono at 24 kHz.
Each instance serves one request at a time; `close()` releases its graph resources.

## Audio streaming

Supply `on_audio` to receive PCM while generation is still running:

```python
from queue import SimpleQueue
from fast_lfm_audio import Pipeline

audio_queue = SimpleQueue()  # A separate playback worker consumes these chunks.
with Pipeline(dtype="fp32") as pipeline:  # "fp16" and "bf16" also supported
    text, waveform, output = pipeline(
        task="chat",
        audio="question.wav",
        on_audio=audio_queue.put,
        chunk_frames=4,
    )
audio_queue.put(None)  # Signal completion to the playback worker.
```

Each callback receives an owned CPU float32 tensor of shape `(1, samples)`.
Four Mimi frames contain 320 ms of audio. The callback should enqueue promptly;
blocking playback inside it also blocks generation. The returned waveform is
exactly the concatenation of the delivered chunks.

Streaming decodes accumulated prefixes and emits new samples. **FP32 streaming
keeps the model and Mimi in FP32, with TF32 disabled.** Text and audio codes retain
their reference bits on the validated inputs. Decoder shape changes introduce
small waveform differences: the largest measured absolute difference was
`6.26e-7` across the short and long requests checked.

For full-waveform bit equality, omit `on_audio` or use
`streaming_mode="buffered"`. Buffered callbacks wait for the complete waveform.
Streaming requires `backend="transformers"` and `codec="fast-mimi"` (the defaults).

Warm representative requests using `on_audio=lambda chunk: None` before live use.
Loading, compilation, and graph capture add first-use latency; new input shapes
can require further capture. Prefix decoding recomputes history and uses extra
GPU memory. Long responses retain the graphs needed for first audio.

## Voice chat latency

RTX 5070 Ti, PyTorch 2.13.0+cu130, Triton 3.7.1, batch one, greedy generation.
Input: a 4.904-second recording. Output: 3.2 seconds of audio with a 64-token budget,
which can truncate the reply. Values are medians of 30 warm requests; p95 uses
nearest rank. FP16/BF16 output samples and codes can differ from FP32.

| Measurement | FP32 full decode | FP16 streaming | BF16 streaming | FP32 streaming |
| --- | ---: | ---: | ---: | ---: |
| First audio (TTFA) | 326.03 ms | 75.10 ms | 74.68 ms | **62.45 ms** |
| TTFA p95 | 327.29 ms | 75.89 ms | 75.25 ms | **63.20 ms** |
| Total generation | **326.31 ms** | 381.75 ms | 380.64 ms | 350.80 ms |
| First-audio speedup | 1x | 4.34x | 4.37x | **5.22x** |

TTFA measures request start to the first owned CPU PCM chunk, including input
preparation, generation, decoding, and CPU transfer. Recording/VAD, networking,
file output, and speaker latency are excluded. Streaming delivers audio earlier
but takes longer to finish the complete waveform. All 30 warm FP32 streaming
requests were below 200 ms. The first streaming request after loading took
1.10 seconds; model loading took a separate 2.82 seconds.

## Precision and runtime

Strict FP32 disables TF32 and autocast, preserving reference reduction order and
rounding boundaries. Full decoding retains the original decoder frame lengths.
Eligible checkpoint weights use lossless integer storage and are reconstructed
bit for bit before FP32 arithmetic. Gradient-enabled calls use the original
PyTorch operations.

| Component | Optimization |
| --- | --- |
| LFM2 backbone | Exact RMSNorm and GEMV, fused gated SiLU, CUDA graphs |
| Depthformer | Exact norm/GEMV fusions, small attention, greedy frame graphs |
| FastConformer encoder | Fused scaled residual additions and CUDA graphs |
| Mimi decoder | Fused scaled residual additions, CUDA graphs, prefix streaming |

The exact GEMV and small-attention kernels are restricted to the validated
RTX 5070 Ti (SM 12.0, 70 SMs), PyTorch `2.13.0+cu130`, Triton `3.7.1`, and
`nvidia-cublas==13.1.1.3`. Other runtimes retain the original matrix and attention
operations. This is not a speed or bit-equality guarantee for arbitrary hardware.
Keep the model's weights, device, and dtype fixed while optimized. The FP32
precision context temporarily changes process-wide PyTorch flags.

FP16/BF16 models use fast-mimi's mixed decoder: BF16 weights and activations with
FP32 residuals. Its atomic reductions can change waveform samples between runs.
Neither reduced-precision mode preserves FP32 output bits.

## vLLM and SGLang

```bash
bash scripts/backend.sh vllm --setup
bash scripts/backend.sh vllm --text 'Hello from LFM.' --output speech.wav

bash scripts/backend.sh sglang --setup
bash scripts/backend.sh sglang --task chat --audio question.wav --output answer.wav
```

These experimental BF16 adapters use separate environments with vLLM 0.29.0 or
SGLang 0.5.19. The LFM2 backbone runs natively; the audio encoder and Depthformer
use the pinned Transformers PR. They support batch one, greedy sampling, and a
maximum 4096-token context. SGLang's setup overrides its declared Transformers
and tokenizers pins to load the audio model, leaving those two dependency conflicts.

For Python use `Pipeline(backend="vllm", dtype="bf16")` or
`Pipeline(backend="sglang", dtype="bf16")` inside the matching environment.
Put native-backend calls under `if __name__ == "__main__":` for worker processes.

Dependencies and adapted source retain their respective licenses;
see [third-party notices](THIRD_PARTY_NOTICES.md).
