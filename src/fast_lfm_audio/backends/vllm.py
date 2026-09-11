"""vLLM's native LFM2 kernels/cache, with audio sampling outside its CUDA graph."""

import os

from vllm import LLM, SamplingParams
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM

from .audio import AudioState
from .plugins import register_vllm


class NativeAudioLfm2(Lfm2ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        source = getattr(self.config, "fast_lfm_audio_source", None)
        self.audio_state = AudioState(source) if source is not None else None

    def embed_input_ids(self, input_ids):
        embeddings = super().embed_input_ids(input_ids)
        return self.audio_state.embed(input_ids, embeddings) if self.audio_state is not None else embeddings

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if self.audio_state is not None and inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        if self.audio_state is not None and self.audio_state.active:
            logits, _ = self.audio_state.step(hidden_states[-1], logits)
        return logits


class AudioWorker:
    def reset_lfm_audio(self, mode):
        self.model_runner.model.audio_state.reset(mode)

    def get_lfm_audio(self):
        return self.model_runner.model.audio_state.events


class VllmEngine:
    def __init__(self, path, max_cache_len, graphs):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        register_vllm()
        self.engine = LLM(
            model=str(path),
            model_impl="vllm",
            dtype="bfloat16",
            max_model_len=max_cache_len,
            max_num_seqs=1,
            kv_cache_memory_bytes=256 * 1024**2,
            gpu_memory_utilization=0.55,
            enable_prompt_embeds=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            async_scheduling=False,
            enforce_eager=not graphs,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
            },
            worker_extension_cls="fast_lfm_audio.backends.vllm.AudioWorker",
        )

    def generate(self, embeddings, mode, max_new_tokens):
        self.engine.collective_rpc("reset_lfm_audio", kwargs={"mode": mode})
        self.engine.generate(
            {"prompt_embeds": embeddings.cpu()},
            SamplingParams(temperature=0, max_tokens=max_new_tokens, detokenize=False),
            use_tqdm=False,
        )
        return self.engine.collective_rpc("get_lfm_audio")[0]

    def close(self):
        self.engine.llm_engine.engine_core.shutdown()
