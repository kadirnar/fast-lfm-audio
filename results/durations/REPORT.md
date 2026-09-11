# LFM2.5 Audio Benchmark

NVIDIA GeForce RTX 5070 Ti; PyTorch 2.13.0+cu130; CUDA 13.0; Python 3.13.14.
Batch one, BF16 model, greedy sampling, one first request and 5 warm repeats per case.
Engines run sequentially in separate processes. Tables show synchronized wall-clock medians; model loading is excluded.
Input preparation, token generation, and full waveform decoding are included. Every request is generated again.
Liquid Audio uses its FP32 LFM detokenizer; fast uses half-precision fast-mimi. Token parity is not waveform parity.

TTS duration cases generate prefixes of longer text: 63/250/1250 frames give 5.04/20/100 s. EOS is unchanged.
They stop at an event budget, not natural sentence completion. The 100 s fixture uses 300 words; an initial 420-word probe returned text instead of speech.
Chat inputs repeat and crop the upstream 4.904 s question to 5/20/100 s, with a 512-event response budget.
This measures synthetic input-length scaling, not natural long-conversation quality. Output durations are independent of input durations.

## End-to-End Latency

| Case | Input s | Output s, base / fast | Liquid s | Fast s | Speedup | Tokens | Budget reached, base / fast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tts_5s | - | 5.04 / 5.04 | 2.219 | 0.513 | 4.33x | exact | True / True |
| tts_20s | - | 20.00 / 20.00 | 8.664 | 2.011 | 4.31x | exact | True / True |
| tts_100s | - | 100.00 / 100.00 | 44.291 | 10.039 | 4.41x | exact | True / True |
| chat_5s | 5 | 13.76 / 13.76 | 6.234 | 1.546 | 4.03x | exact | False / False |
| chat_20s | 20 | 36.88 / 36.88 | 16.230 | 3.918 | 4.14x | exact | True / True |
| chat_100s | 100 | 21.60 / 21.60 | 9.885 | 2.526 | 3.91x | exact | False / False |

A budget-limited output is partial. One event is a text token or an eight-codebook audio frame.

**100 s TTS warning:** the optimized output has a 70 s low-signal tail. This is a compute stress test, not 100 s of successful continuous speech. Compare the reference below.

## Timing Breakdown

| Case / engine | Prepare ms | Generate ms | Decode ms | First audio token ms | First request s | p95 total s | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tts_5s / liquid | 1.0 | 2213.1 | 4.9 | 51.9 | 2.762 | 2.221 | 3.17 |
| tts_5s / fast | 0.8 | 511.3 | 0.8 | 17.8 | 1.228 | 0.514 | 3.79 |
| tts_20s / liquid | 1.2 | 8653.3 | 9.5 | 51.9 | 8.636 | 8.681 | 3.23 |
| tts_20s / fast | 1.0 | 2007.7 | 2.2 | 18.4 | 2.092 | 2.014 | 4.20 |
| tts_100s / liquid | 1.5 | 44207.9 | 81.9 | 55.2 | 43.224 | 45.500 | 4.16 |
| tts_100s / fast | 1.5 | 10022.1 | 15.4 | 21.4 | 10.288 | 10.045 | 7.37 |
| chat_5s / liquid | 2.3 | 6224.3 | 7.2 | 101.6 | 6.445 | 6.476 | 3.22 |
| chat_5s / fast | 2.3 | 1541.5 | 2.2 | 54.5 | 1.806 | 1.549 | 7.37 |
| chat_20s / liquid | 2.8 | 16208.4 | 19.1 | 102.7 | 16.175 | 16.272 | 3.32 |
| chat_20s / fast | 3.6 | 3909.7 | 4.1 | 55.8 | 3.964 | 3.921 | 8.14 |
| chat_100s / liquid | 5.9 | 9868.8 | 10.2 | 138.2 | 9.918 | 9.928 | 3.48 |
| chat_100s / fast | 10.2 | 2511.0 | 4.1 | 93.1 | 2.534 | 2.529 | 8.39 |

First audio token is not first playable PCM audio: this pipeline decodes the complete output offline.
First-request timing includes graph capture, but reuses existing Triton caches and any earlier shapes in that process.
Model setup (s): liquid 2.100, fast 1.536.

## Waveform Checks

| Case / decoder | Low-signal seconds | Trailing low-signal seconds | Finite |
| --- | --- | --- | --- |
| tts_5s / liquid | 0 | 0 | True |
| tts_5s / fast | 0 | 0 | True |
| tts_5s / mimi_fp32 | 0 | 0 | True |
| tts_20s / liquid | 0 | 0 | True |
| tts_20s / fast | 0 | 0 | True |
| tts_20s / mimi_fp32 | 0 | 0 | True |
| tts_100s / liquid | 71 | 70 | True |
| tts_100s / fast | 71 | 70 | True |
| tts_100s / mimi_fp32 | 71 | 70 | True |
| chat_5s / liquid | 4 | 0 | True |
| chat_5s / fast | 4 | 0 | True |
| chat_5s / mimi_fp32 | 4 | 0 | True |
| chat_20s / liquid | 3 | 0 | True |
| chat_20s / fast | 17 | 6 | True |
| chat_20s / mimi_fp32 | 17 | 6 | True |
| chat_100s / liquid | 3 | 0 | True |
| chat_100s / fast | 3 | 0 | True |
| chat_100s / mimi_fp32 | 3 | 0 | True |

Low-signal: RMS < 0.001 (-60 dBFS) in full one-second windows of saved PCM16 WAVs. This is not an intelligibility score.
Native FP32 Mimi decodes the same reference codes, separating codec differences from optimization error. Saved checks are reused when WAVs are absent.

## Same-Code Decoders

| Case | LFM ms | Mimi FP32 ms | Fast Mimi ms | vs LFM | vs Mimi | SNR dB |
| --- | --- | --- | --- | --- | --- | --- |
| tts_5s | 4.35 | 6.45 | 0.65 | 6.73x | 9.99x | 48.2 |
| tts_20s | 9.46 | 14.45 | 2.05 | 4.61x | 7.04x | 47.4 |
| tts_100s | 87.46 | 69.06 | 15.26 | 5.73x | 4.53x | 50.6 |
| chat_5s | 6.68 | 11.13 | 2.02 | 3.30x | 5.50x | 47.2 |
| chat_20s | 19.82 | 24.99 | 3.92 | 5.05x | 6.37x | 46.7 |
| chat_100s | 10.19 | 15.48 | 3.94 | 2.59x | 3.93x | 47.4 |

Medians of 20 warm runs on identical codes. SNR compares fast-mimi with native FP32 Mimi; it is not a listening test.

## Initial Long-Shape Tuning

The initial 100 s probe took 277.768 s, including 266.753 s of decode/autotuning for the 2048-frame bucket.
Smaller-shape caches already existed. Final warm medians exclude this cost. Prewarm required shapes before serving requests.

## Sources and Artifacts

- transformers: `843101f38c800b98d49b704215f0b76b92e48e64`
- fast-mimi: `f6825daa223d2851d6918561f5fde02cf28d6d9e`
- liquid-audio: `19e65845923a7f136442c95137884ec61eb386aa`
- Model: `c362a0625dfe45aa588dce5f0ada28a7e5707628`

Engine JSON files contain every timing sample, requests, metadata, memory measurements, and token hashes.
`parity.json` records whether comparisons used local tensors or published SHA256 hashes. `waveforms.json` records signal checks.
WAV/PT artifacts are generated locally and are not committed. Reproduce with `python -m benchmarks.run`; add `--suite durations` for 5/20/100 s tests.
