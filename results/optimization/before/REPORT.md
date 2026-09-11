# Three-Backend Benchmark

NVIDIA GeForce RTX 5070 Ti; PyTorch 2.13.0+cu130; CUDA 13.0; BF16; batch one.
Each case: one first request, then 5 warm requests. Tables show warm medians in seconds.
End-to-end = input preparation + model generation + complete fast-mimi decoding, with CUDA synchronization.
Includes native-engine IPC. Excludes model loading, initial setup, file writing, recording, and playback.
All engines run separately. This is an offline, single-request comparison, not a serving-throughput test.

Transformers uses this repo's backbone/depthformer CUDA graphs. The other columns use this repo's
experimental adapters: native LFM2 backbone kernels and caches, the PR's audio encoder/depthformer, and fast-mimi.
They are not upstream native LFM2-Audio implementations, and never fall back to a Transformers backbone.

## Versions

| Backend | Library | Python | Model Setup (s) |
| --- | --- | --- | --- |
| Transformers | 5.16.0.dev0 | 3.13.14 | 1.67 |
| vLLM | 0.29.0 | 3.13.14 | 17.83 |
| SGLang | 0.5.19 | 3.12.13 | 21.14 |

SGLang 0.5.19 pins Transformers 5.12.1 and tokenizers 0.22.2. This adapter explicitly overrides
them with the pinned audio PR (5.16.0.dev0) and tokenizers 0.23.2, with a narrow config-registration shim.
Its environment therefore has two declared dependency conflicts despite passing these inference tests.
Native decoding uses CUDA graphs; vLLM uses FlashAttention 2 and SGLang uses FA4 on this RTX 5070 Ti.
The isolated JIT toolkit is CUDA 13.4; Torch's runtime stays CUDA 13.0. No upstream source files are edited.

## Text to Speech

| Audio Budget | Transformers | vLLM | SGLang |
| --- | --- | --- | --- |
| 5 s | **0.515 s**<br><sub>5.04 s audio (capped)</sub> | **0.578 s**<br><sub>5.04 s audio (capped)</sub> | **0.555 s**<br><sub>5.04 s audio (capped)</sub> |
| 20 s | **2.012 s**<br><sub>20.00 s audio (capped)</sub> | **2.259 s**<br><sub>20.00 s audio (capped)</sub> | **2.133 s**<br><sub>20.00 s audio (capped)</sub> |
| 100 s | **9.949 s**<br><sub>100.00 s audio (capped)</sub> | **11.264 s**<br><sub>100.00 s audio (capped)</sub> | **10.550 s**<br><sub>100.00 s audio (capped)</sub> |

## Voice Chat

| Input Audio | Transformers | vLLM | SGLang |
| --- | --- | --- | --- |
| 5 s | **1.529 s**<br><sub>13.76 s audio</sub> | **1.406 s**<br><sub>10.72 s audio</sub> | **4.083 s**<br><sub>36.80 s audio (capped)</sub> |
| 20 s | **3.881 s**<br><sub>36.88 s audio (capped)</sub> | **4.382 s**<br><sub>36.32 s audio (capped)</sub> | **4.158 s**<br><sub>37.12 s audio (capped)</sub> |
| 100 s | **2.506 s**<br><sub>21.60 s audio</sub> | **4.229 s**<br><sub>32.64 s audio (capped)</sub> | **2.795 s**<br><sub>19.76 s audio</sub> |

**TTS uses fixed event budgets, not sentence completion.** 63/250/1250 frames produce 5.04/20/100 s.
**The 100 s case is a stress test, not 100 s of continuous speech.** Long low-signal tails occur in all three outputs.
Chat inputs tile/crop the upstream 4.904 s recording; these are synthetic length tests, not natural long conversations.
Chat replies differ in content/length across backends. Raw chat latency is not an equal-output speedup comparison.
Capped replies may be incomplete. Token equality is not a perceptual quality test.

## Exact Tokens Across Repeats

| Case | Transformers | vLLM | SGLang |
| --- | --- | --- | --- |
| tts_5s | yes | yes | yes |
| tts_20s | yes | yes | yes |
| tts_100s | yes | yes | yes |
| chat_5s | yes | yes | yes |
| chat_20s | yes | yes | yes |
| chat_100s | yes | yes | yes |

## Exact Tokens vs Transformers

| Case | vLLM | SGLang |
| --- | --- | --- |
| tts_5s | no | no |
| tts_20s | no | no |
| tts_100s | no | no |
| chat_5s | no | no |
| chat_20s | no | no |
| chat_100s | no | no |

## 100 s Waveform Audit

| Backend | Trailing Quiet Audio | Finite Samples |
| --- | --- | --- |
| Transformers | 70 s | True |
| vLLM | 79 s | True |
| SGLang | 79 s | True |

Quiet = one-second RMS below 0.001. Audio lengths and quiet tails do not establish intelligibility.
Raw measurements: [Transformers](fast.json), [vLLM](vllm.json), [SGLang](sglang.json).
First-request timings are in each JSON; disk JIT caches may already be warm.
JSON memory counters cover the driver process only, not native engine workers; they are not total-engine VRAM.
Validation: [token parity](parity.json), [waveform statistics](waveforms.json).

Native implementations: [vLLM LFM2](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/lfm2.py),
[SGLang LFM2](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/models/lfm2.py).
