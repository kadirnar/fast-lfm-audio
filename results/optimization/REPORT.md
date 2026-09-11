# Runtime Optimization

Same GPU, model weights, requests, greedy sampling, context limits, and repeat counts.
End-to-end medians include input preparation, generation, full fast-mimi decoding, and CUDA synchronization.
Model loading and first-use setup are excluded. Speedups are shown only for exact token matches.
The shared greedy Depthformer now uses selective torch.compile fusion before CUDA graph capture.
ATen mean reductions, eager rotary-buffer initialization, and BF16 rounding points are preserved.
Weights, codebook counts, generation budgets, and sampling rules are unchanged; no quantization is used.
First use includes compilation; disk compiler caches may already be warm in these measurements.

Saved PCM files: 9/18 bitwise equal.
Token equality does not imply bitwise PCM equality. [Waveform comparisons](waveform-parity.json).

## Transformers

| Case | Before | After | Speedup | Exact Tokens |
| --- | --- | --- | --- | --- |
| tts_5s | 0.515 s | 0.406 s | 1.27x | yes |
| tts_20s | 2.012 s | 1.587 s | 1.27x | yes |
| tts_100s | 9.949 s | 7.907 s | 1.26x | yes |
| chat_5s | 1.529 s | 1.248 s | 1.22x | yes |
| chat_20s | 3.881 s | 3.126 s | 1.24x | yes |
| chat_100s | 2.506 s | 2.059 s | 1.22x | yes |

## vLLM

| Case | Before | After | Speedup | Exact Tokens |
| --- | --- | --- | --- | --- |
| tts_5s | 0.578 s | 0.477 s | 1.21x | yes |
| tts_20s | 2.259 s | 1.854 s | 1.22x | yes |
| tts_100s | 11.264 s | 9.248 s | 1.22x | yes |
| chat_5s | 1.406 s | 1.186 s | 1.19x | yes |
| chat_20s | 4.382 s | 3.631 s | 1.21x | yes |
| chat_100s | 4.229 s | 3.548 s | 1.19x | yes |

## SGLang

| Case | Before | After | Speedup | Exact Tokens |
| --- | --- | --- | --- | --- |
| tts_5s | 0.555 s | 0.454 s | 1.22x | yes |
| tts_20s | 2.133 s | 1.725 s | 1.24x | yes |
| tts_100s | 10.550 s | 8.519 s | 1.24x | yes |
| chat_5s | 4.083 s | 3.332 s | 1.23x | yes |
| chat_20s | 4.158 s | 3.399 s | 1.22x | yes |
| chat_100s | 2.795 s | 2.404 s | 1.16x | yes |

The 100 s TTS case contains long quiet tails and is a fixed-budget stress test, not continuous speech.
Chat input audio is repeated/cropped from the upstream fixture. Some responses hit the token limit.
Before/after token equality checks both content and length; it does not establish perceptual quality.

[Before](before/REPORT.md) | [After](after/REPORT.md) | [Token checks](parity.json)

## Generation Profile

| Stage | Before (ms) | After (ms) |
| --- | --- | --- |
| input_embeddings | 0.09 | 0.13 |
| llm_prefill | 9.22 | 9.51 |
| llm_decode | 215.18 | 213.34 |
| depthformer | 281.28 | 175.46 |

Profile: Transformers, tts_5s. CUDA-event stream intervals include launch/idle gaps.
Instrumentation adds overhead; use the uninstrumented tables above for speed comparisons.
[Before profile](profile-before.json) | [After profile](profile-after.json)

## Profiled Kernel Calls

| Before | After |
| --- | --- |
| 200934 | 123192 |

## Reproduce

```bash
FAST_LFM_COMPILE_DEPTH=0 python -m benchmarks.compare_backends --output-dir results/optimization/before
python -m benchmarks.compare_backends --output-dir results/optimization/after
FAST_LFM_COMPILE_DEPTH=0 python -m benchmarks.profile --output results/optimization/profile-before.json
python -m benchmarks.profile --output results/optimization/profile-after.json
python -m benchmarks.optimization_report
```

Set `FAST_LFM_COMPILE_DEPTH=0` to use the previous graph-only depth path.
Compiler API: [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile).
