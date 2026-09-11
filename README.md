# Fast LFM Audio

LFM2.5 Audio 1.5B inference with CUDA graphs and
[fast-mimi](https://github.com/kadirnar/fast-mimi). TTS, spoken chat, and ASR through
[Transformers PR #48249](https://github.com/huggingface/transformers/pull/48249).
Benchmarks compare against [Liquid Audio](https://github.com/Liquid4All/liquid-audio).

## Setup

Requires Linux, an NVIDIA CUDA GPU, `git`, and `uv`.
Tested on RTX 5070 Ti with Python 3.13, PyTorch 2.13.0 and CUDA 13.0.

```bash
git clone https://github.com/kadirnar/fast-lfm-audio.git
cd fast-lfm-audio
bash scripts/setup.sh
source .venv/bin/activate
```

The setup script pins the exact Transformers PR commit, fast-mimi, and Liquid
Audio in `vendor/`. Python dependencies are pinned in `requirements-lock.txt`.
Use this setup instead of installing stock Transformers from PyPI.

## Run

```bash
# Text to speech
fast-lfm-audio --text 'Hello, this is a test.' --output speech.wav

# Spoken answer to text
fast-lfm-audio --task chat --text 'Tell me about the moon.' --output answer.wav

# Spoken answer to audio
fast-lfm-audio --task chat --audio question.wav --output answer.wav

# Transcription
fast-lfm-audio --task asr --audio question.wav
```

TTS voices: `US male`, `US female`, `UK male`, `UK female` (default).
Select with `--voice 'US male'`. Audio is mono, 24 kHz.
Use `--help` for generation limits and decoder options.

## Python

```python
import soundfile as sf
from fast_lfm_audio import Pipeline

pipeline = Pipeline()
inputs = pipeline.prepare(text="Hello from LFM.")
text, audio, output = pipeline.generate(inputs, max_new_tokens=512, audio_top_k=1)
sf.write("speech.wav", audio[0].cpu().numpy(), 24000)

inputs = pipeline.prepare(task="chat", audio="question.wav")
text, audio, output = pipeline.generate(
    inputs, generation_mode="interleaved", max_new_tokens=512, text_top_k=1, audio_top_k=1
)
```

Reuse the same `Pipeline` to reuse captured graphs. Existing Transformers code
can use `handle = optimize(model.eval())` from `fast_lfm_audio`; call
`handle.close()` to restore the original methods.

## Benchmarks

```bash
# Four TTS voices, text chat, and audio chat
python -m benchmarks.run

# 5 / 20 / 100-second workloads
python -m benchmarks.run --suite durations

# Also compare unoptimized Transformers and depth-only graphs
python -m benchmarks.run --engines liquid hf depth fast

# Rebuild the report without running inference
python -m benchmarks.run --suite durations --report-only

pytest -q
```

Default: one first request and five warm repeats, batch one, greedy sampling.
GPU processes run sequentially. Reports separate preparation, generation,
decoding, first audio token, cold latency, memory, and token parity.

### End-to-End Latency

Measured on RTX 5070 Ti: median of five warm requests, batch one, greedy sampling.
End-to-end includes input preparation, token generation, and full waveform
decoding. Model loading, first-use graph capture/kernel tuning, and audio
recording/playback are excluded. These are complete-output latencies, not time
to first audio.

| Task | Input audio (s) | Output audio (s) | Liquid Audio E2E (s) | Fast LFM Audio E2E (s) | Speedup |
| --- | --- | --- | --- | --- | --- |
| TTS | - | 5.04 | 2.219 | 0.513 | 4.33x |
| TTS | - | 20.00 | 8.664 | 2.011 | 4.31x |
| TTS (stress test) | - | 100.00 | 44.291 | 10.039 | 4.41x |
| Voice chat | 5 | 13.76 | 6.234 | 1.546 | 4.03x |
| Voice chat | 20 | 36.88 | 16.230 | 3.918 | 4.14x |
| Voice chat | 100 | 21.60 | 9.885 | 2.526 | 3.91x |

TTS rows are duration-capped prefixes, not naturally completed utterances; the
5-second target produces 5.04 seconds. Chat inputs repeat/crop the same 4.904-second
recording. Output lengths vary; the 20-second chat case reaches its 512-event
limit, so that response is partial.

**100-second TTS warning:** both engines produce a roughly 70-second low-signal
tail. This is a compute stress test, not 100 seconds of successful continuous speech.

- [Standard results](results/REPORT.md): about 4x faster on the measured cases.
- [5/20/100-second results](results/durations/REPORT.md): TTS output lengths and
  repeated-speech chat input lengths, with waveform checks.

Raw JSON measurements are committed. WAVs, tensors, model weights, downloaded
repositories, and local experiments are excluded from Git.

## Limits

- CUDA, batch one, one request at a time. LFM weights remain BF16, without quantization.
- Output is decoded offline, not streamed. First audio token is not first playable audio.
- First use captures graphs and may autotune kernels. The first 100-second Mimi
  decode/autotuning took about 267 seconds; warm timings exclude that cost.
- The 100-second TTS fixture produced a roughly 70-second low-signal tail in both
  engines. It is a stress test, not successful 100-second continuous speech.
- Fast-mimi and the default LFM detokenizer produce different waveforms. Exact
  token parity and Mimi SNR do not establish perceptual speech quality.
- Default cache capacity: 2048 prompt-plus-output events. Use `--max-cache-len 4096`
  for longer contexts. Greedy parity is verified only on the pinned, tested setup.

## Code

- `src/fast_lfm_audio/runtime.py`: depthformer and backbone CUDA graphs.
- `src/fast_lfm_audio/pipeline.py`: model loading, input preparation, and Mimi decoding.
- `src/fast_lfm_audio/cli.py`: command-line interface.
- `benchmarks/`: shared cases, isolated runners, decoder measurements, and reports.
