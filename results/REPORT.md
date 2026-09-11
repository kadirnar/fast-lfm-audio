# LFM2.5 Audio Benchmark

NVIDIA GeForce RTX 5070 Ti; PyTorch 2.13.0+cu130; CUDA 13.0; Python 3.13.14.
Batch one, BF16 model, greedy sampling, one first request and 5 warm repeats per case.
Engines run sequentially in separate processes. Tables show synchronized wall-clock medians; model loading is excluded.
Input preparation, token generation, and full waveform decoding are included. Every request is generated again.
Liquid Audio uses its FP32 LFM detokenizer; fast uses half-precision fast-mimi. Token parity is not waveform parity.

## End-to-End Latency

| Case | Input s | Output s, base / fast | Liquid s | Fast s | Speedup | Tokens | Budget reached, base / fast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tts_uk_female | - | 3.68 / 3.68 | 1.647 | 0.392 | 4.20x | exact | False / False |
| tts_us_male | - | 4.96 / 4.96 | 2.199 | 0.522 | 4.22x | exact | False / False |
| tts_us_female | - | 4.32 / 4.32 | 1.942 | 0.456 | 4.26x | exact | False / False |
| tts_uk_male | - | 6.16 / 6.16 | 2.720 | 0.642 | 4.24x | exact | False / False |
| chat_text | - | 8.08 / 8.08 | 3.776 | 0.940 | 4.02x | exact | False / False |
| chat_audio | - | 21.68 / 21.68 | 9.859 | 2.436 | 4.05x | exact | False / False |

A budget-limited output is partial. One event is a text token or an eight-codebook audio frame.

## Timing Breakdown

| Case / engine | Prepare ms | Generate ms | Decode ms | First audio token ms | First request s | p95 total s | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tts_uk_female / liquid | 0.9 | 1641.2 | 4.9 | 51.1 | 2.206 | 1.651 | 3.17 |
| tts_uk_female / hf | 0.7 | 1625.5 | 5.4 | 43.2 | 2.212 | 1.645 | 3.04 |
| tts_uk_female / depth | 0.7 | 594.3 | 5.2 | 21.5 | 1.318 | 0.602 | 3.08 |
| tts_uk_female / fast | 0.7 | 390.4 | 0.8 | 17.7 | 1.086 | 0.393 | 3.76 |
| tts_us_male / liquid | 0.9 | 2193.2 | 4.9 | 51.2 | 2.188 | 2.203 | 3.17 |
| tts_us_male / hf | 0.7 | 2220.6 | 5.5 | 43.4 | 2.206 | 2.280 | 3.05 |
| tts_us_male / depth | 0.7 | 789.8 | 5.2 | 21.1 | 0.801 | 0.798 | 3.08 |
| tts_us_male / fast | 0.7 | 520.0 | 0.8 | 18.3 | 0.521 | 0.529 | 3.76 |
| tts_us_female / liquid | 0.9 | 1936.7 | 4.9 | 51.6 | 1.925 | 1.945 | 3.17 |
| tts_us_female / hf | 0.8 | 1952.1 | 5.4 | 43.8 | 1.932 | 1.961 | 3.05 |
| tts_us_female / depth | 0.7 | 692.3 | 5.2 | 21.2 | 0.699 | 0.699 | 3.08 |
| tts_us_female / fast | 0.6 | 455.1 | 0.8 | 17.7 | 0.458 | 0.458 | 3.76 |
| tts_uk_male / liquid | 0.9 | 2713.9 | 4.8 | 51.2 | 2.722 | 2.720 | 3.17 |
| tts_uk_male / hf | 0.7 | 2706.5 | 5.3 | 43.7 | 2.745 | 2.724 | 3.05 |
| tts_uk_male / depth | 0.7 | 977.0 | 5.2 | 21.2 | 0.987 | 0.984 | 3.08 |
| tts_uk_male / fast | 0.7 | 639.7 | 1.3 | 17.9 | 0.658 | 0.645 | 3.95 |
| chat_text / liquid | 0.9 | 3770.0 | 4.9 | 92.7 | 3.771 | 3.790 | 3.18 |
| chat_text / hf | 0.7 | 3790.5 | 5.3 | 84.1 | 3.710 | 4.014 | 3.06 |
| chat_text / depth | 0.7 | 1487.9 | 5.3 | 61.7 | 1.501 | 1.499 | 3.09 |
| chat_text / fast | 0.8 | 938.3 | 1.3 | 37.4 | 0.940 | 0.942 | 3.95 |
| chat_audio / liquid | 2.2 | 9846.4 | 10.3 | 101.7 | 9.958 | 9.981 | 3.23 |
| chat_audio / hf | 2.2 | 9608.3 | 10.7 | 100.1 | 10.506 | 9.678 | 3.11 |
| chat_audio / depth | 2.1 | 3810.3 | 10.8 | 78.8 | 4.027 | 3.829 | 3.15 |
| chat_audio / fast | 2.4 | 2429.0 | 4.1 | 54.4 | 2.721 | 2.437 | 4.75 |

First audio token is not first playable PCM audio: this pipeline decodes the complete output offline.
First-request timing includes graph capture, but reuses existing Triton caches and any earlier shapes in that process.
Model setup (s): liquid 2.022, hf 1.388, depth 1.257, fast 1.527.

## Waveform Checks

| Case / decoder | Low-signal seconds | Trailing low-signal seconds | Finite |
| --- | --- | --- | --- |
| tts_uk_female / liquid | 0 | 0 | True |
| tts_uk_female / hf | 0 | 0 | True |
| tts_uk_female / depth | 0 | 0 | True |
| tts_uk_female / fast | 0 | 0 | True |
| tts_uk_female / mimi_fp32 | 0 | 0 | True |
| tts_us_male / liquid | 0 | 0 | True |
| tts_us_male / hf | 0 | 0 | True |
| tts_us_male / depth | 0 | 0 | True |
| tts_us_male / fast | 0 | 0 | True |
| tts_us_male / mimi_fp32 | 0 | 0 | True |
| tts_us_female / liquid | 0 | 0 | True |
| tts_us_female / hf | 0 | 0 | True |
| tts_us_female / depth | 0 | 0 | True |
| tts_us_female / fast | 0 | 0 | True |
| tts_us_female / mimi_fp32 | 0 | 0 | True |
| tts_uk_male / liquid | 0 | 0 | True |
| tts_uk_male / hf | 0 | 0 | True |
| tts_uk_male / depth | 0 | 0 | True |
| tts_uk_male / fast | 0 | 0 | True |
| tts_uk_male / mimi_fp32 | 0 | 0 | True |
| chat_text / liquid | 0 | 0 | True |
| chat_text / hf | 0 | 0 | True |
| chat_text / depth | 0 | 0 | True |
| chat_text / fast | 0 | 0 | True |
| chat_text / mimi_fp32 | 0 | 0 | True |
| chat_audio / liquid | 4 | 0 | True |
| chat_audio / hf | 4 | 0 | True |
| chat_audio / depth | 4 | 0 | True |
| chat_audio / fast | 4 | 0 | True |
| chat_audio / mimi_fp32 | 4 | 0 | True |

Low-signal: RMS < 0.001 (-60 dBFS) in full one-second windows of saved PCM16 WAVs. This is not an intelligibility score.
Native FP32 Mimi decodes the same reference codes, separating codec differences from optimization error. Saved checks are reused when WAVs are absent.

## Same-Code Decoders

| Case | LFM ms | Mimi FP32 ms | Fast Mimi ms | vs LFM | vs Mimi | SNR dB |
| --- | --- | --- | --- | --- | --- | --- |
| tts_uk_female | 4.43 | 6.32 | 0.65 | 6.84x | 9.77x | 46.1 |
| tts_us_male | 4.85 | 6.94 | 0.65 | 7.50x | 10.73x | 47.5 |
| tts_us_female | 4.33 | 6.73 | 0.65 | 6.69x | 10.40x | 46.0 |
| tts_uk_male | 4.43 | 7.33 | 1.16 | 3.83x | 6.33x | 49.8 |
| chat_text | 4.64 | 8.38 | 1.16 | 4.00x | 7.21x | 46.8 |
| chat_audio | 10.18 | 15.57 | 4.17 | 2.44x | 3.74x | 47.8 |

Medians of 20 warm runs on identical codes. SNR compares fast-mimi with native FP32 Mimi; it is not a listening test.

## Sources and Artifacts

- transformers: `843101f38c800b98d49b704215f0b76b92e48e64`
- fast-mimi: `f6825daa223d2851d6918561f5fde02cf28d6d9e`
- liquid-audio: `19e65845923a7f136442c95137884ec61eb386aa`
- Model: `c362a0625dfe45aa588dce5f0ada28a7e5707628`

Engine JSON files contain every timing sample, requests, metadata, memory measurements, and token hashes.
`parity.json` records whether comparisons used local tensors or published SHA256 hashes. `waveforms.json` records signal checks.
WAV/PT artifacts are generated locally and are not committed. Reproduce with `python -m benchmarks.run`; add `--suite durations` for 5/20/100 s tests.
