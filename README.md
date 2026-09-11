# Fast LFM Audio

LFM2.5 Audio 1.5B inference for TTS, voice chat, and ASR.
Uses [Transformers PR #48249](https://github.com/huggingface/transformers/pull/48249),
CUDA graphs, and [fast-mimi](https://github.com/kadirnar/fast-mimi).

## Setup

Requires Linux, an NVIDIA CUDA GPU, `git`, and `uv`.

```bash
git clone https://github.com/kadirnar/fast-lfm-audio.git
cd fast-lfm-audio
bash scripts/setup.sh
source .venv/bin/activate
```

Setup installs pinned versions of the Transformers PR, fast-mimi, and Liquid Audio.

## Run

```bash
# Text to speech
fast-lfm-audio --text 'Hello, this is a test.' --output speech.wav

# Spoken answer to audio
fast-lfm-audio --task chat --audio question.wav --output answer.wav

# Transcription
fast-lfm-audio --task asr --audio question.wav
```

Chat also accepts `--text`. TTS voices: `UK female` (default), `UK male`,
`US female`, `US male`; select with `--voice`. More options: `fast-lfm-audio --help`.

### vLLM and SGLang

```bash
bash scripts/backend.sh vllm --setup
bash scripts/backend.sh sglang --setup

bash scripts/backend.sh vllm --text 'Hello from LFM.' --output speech.wav
bash scripts/backend.sh sglang --task chat --audio question.wav --output answer.wav
```

Official vLLM 0.29.0 and SGLang 0.5.19, in separate environments, with **experimental
local audio adapters**. The LFM2 backbone runs natively; the encoder/depthformer
use the Transformers PR. Batch one, greedy sampling, maximum 4096-token context.
SGLang overrides its Transformers/tokenizers pins; [compatibility details](results/backends/inference/REPORT.md#versions).

## Python

```python
import soundfile as sf
from fast_lfm_audio import Pipeline

if __name__ == "__main__":
    with Pipeline(backend="transformers") as model:
        text, audio, _ = model(text="Hello from LFM.")
        sf.write("speech.wav", audio[0].cpu().numpy(), 24000)
        text, audio, _ = model(task="chat", audio="question.wav")
```

Reuse the instance. Audio is mono, 24 kHz. Native backends use the same API
with `backend="vllm"` or `"sglang"` in their matching environment. Keep the
`__main__` guard for their worker processes.

## End-to-End Latency

RTX 5070 Ti, PyTorch 2.13.0, BF16, batch one. Median of five warm requests.
All three use CUDA graphs and fast-mimi. **Transformers is the optimized version in this repo.**

Selective Depthformer fusion reduces total latency by **14-21%** versus the previous
graph-only version, with identical tokens in all 18 cases. [Before / after](results/optimization/REPORT.md).

**Processing time:** input preparation + generation + full audio decoding.
Lower is better. Model loading, first-use setup, file writing, and recording/playback are excluded.

### Text to Speech

| Generated Audio | Transformers | vLLM | SGLang |
| ---: | ---: | ---: | ---: |
| 5.04 s | **0.406 s** | 0.477 s | 0.454 s |
| 20.00 s | **1.587 s** | 1.854 s | 1.725 s |
| 100.00 s* | **7.907 s** | 9.248 s | 8.519 s |

TTS stops at a frame budget, not sentence completion. **100 s is a stress test:**
outputs contain 70-79 s low-signal tails, not 100 s of continuous speech.

### Voice Chat

Each cell shows **processing time**, then reply length. `*` = capped, potentially incomplete reply.

| Input Audio | Transformers | vLLM | SGLang |
| ---: | ---: | ---: | ---: |
| 5 s | **1.248 s**<br><sub>13.76 s reply</sub> | **1.186 s**<br><sub>10.72 s reply</sub> | **3.332 s**<br><sub>36.80 s reply*</sub> |
| 20 s | **3.126 s**<br><sub>36.88 s reply*</sub> | **3.631 s**<br><sub>36.32 s reply*</sub> | **3.399 s**<br><sub>37.12 s reply*</sub> |
| 100 s | **2.059 s**<br><sub>21.60 s reply</sub> | **3.548 s**<br><sub>32.64 s reply*</sub> | **2.404 s**<br><sub>19.76 s reply</sub> |

Inputs repeat/crop a 4.904 s recording. Different replies mean these chat times
are **not an equal-output speed comparison**. Native tokens differ from Transformers.

[Full three-backend results](results/optimization/after/REPORT.md) |
[Earlier Liquid Audio comparison](results/durations/REPORT.md)

## Benchmark

```bash
python -m benchmarks.compare_backends --output-dir results/optimization/after
python -m benchmarks.profile
python -m benchmarks.run --suite durations  # Liquid Audio vs optimized Transformers
pytest -q
```

## Notes

- Single-request, offline inference; no batching, streaming server, or quantization.
- Greedy Depthformer operations are fused with `torch.compile` before CUDA graph capture.
  First use includes compilation; set `FAST_LFM_COMPILE_DEPTH=0` to disable it.
- First-use setup can be slow: the initial 100-second Mimi decode/tuning took 267 seconds.
- Token parity does not establish speech quality or bitwise PCM equality.
