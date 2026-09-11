# Fast LFM Audio

Fast LFM2.5 Audio 1.5B inference for TTS, voice chat, and ASR.
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

## Python

```python
import soundfile as sf
from fast_lfm_audio import Pipeline

model = Pipeline()
text, audio, _ = model(text="Hello from LFM.")
sf.write("speech.wav", audio[0].cpu().numpy(), 24000)

text, audio, _ = model(task="chat", audio="question.wav")
```

Reuse the same instance. Audio is mono, 24 kHz; requests run one at a time.

## End-to-End Latency

Compared with [Liquid Audio](https://github.com/Liquid4All/liquid-audio) on RTX 5070 Ti
(PyTorch 2.13.0, CUDA 13.0). Median of five warm requests, batch one, greedy sampling.
Includes input preparation, generation, and full audio decoding. Excludes model
loading, first-use setup, and recording/playback; this is not time to first audio.

| Task | Input audio (s) | Output audio (s) | Liquid Audio E2E (s) | Fast LFM Audio E2E (s) | Speedup |
| --- | --- | --- | --- | --- | --- |
| TTS | - | 5.04 | 2.219 | 0.513 | 4.33x |
| TTS | - | 20.00 | 8.664 | 2.011 | 4.31x |
| TTS (stress test) | - | 100.00 | 44.291 | 10.039 | 4.41x |
| Voice chat | 5 | 13.76 | 6.234 | 1.546 | 4.03x |
| Voice chat | 20 | 36.88 | 16.230 | 3.918 | 4.14x |
| Voice chat | 100 | 21.60 | 9.885 | 2.526 | 3.91x |

TTS outputs stop at a frame budget, not sentence completion. Chat inputs repeat
and crop a 4.904-second recording; the 20-second case hits its response limit.

**100-second TTS is a stress test:** both engines have a roughly 70-second
low-signal tail, not 100 seconds of continuous speech.

Full results: [standard](results/REPORT.md) and [5/20/100 seconds](results/durations/REPORT.md).

```bash
python -m benchmarks.run                    # Standard cases
python -m benchmarks.run --suite durations  # 5 / 20 / 100 seconds
pytest -q
```

## Notes

- BF16 model, no quantization. Full output is decoded offline, not streamed.
- First-use setup can be slow: the initial 100-second Mimi decode/tuning took 267 seconds.
- Matching tokens do not mean matching waveforms or speech quality; the decoders differ.
