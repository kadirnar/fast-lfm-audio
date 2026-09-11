"""SGLang's native LFM2 cache and kernels with an out-of-graph audio sampler."""

import os

from .compat import configure_sglang


class SGLangEngine:
    def __init__(self, path, max_cache_len, graphs):
        configure_sglang()
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "fast_lfm_audio.backends.sglang_models"
        from sglang import Engine

        self.engine = Engine(
            model_path=str(path),
            model_impl="sglang",
            dtype="bfloat16",
            context_length=max_cache_len,
            max_running_requests=1,
            max_total_tokens=max_cache_len,
            mem_fraction_static=0.55,
            disable_cuda_graph=not graphs,
            cuda_graph_bs_decode=[1],
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            chunked_prefill_size=-1,
            sampling_backend="pytorch",
            attention_backend="fa4",
        )

    def generate(self, embeddings, mode, max_new_tokens):
        from sglang.srt.managers.io_struct import GenerateReqInput

        self.engine.collective_rpc("reset_lfm_audio", mode=mode)
        request = GenerateReqInput(
            input_embeds=embeddings.float().tolist(),
            sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
        )
        generator = self.engine.tokenizer_manager.generate_request(request, None)
        try:
            output = self.engine.loop.run_until_complete(generator.__anext__())
        finally:
            self.engine.loop.run_until_complete(generator.aclose())
        return output["meta_info"]["lfm_audio_events"]

    def close(self):
        self.engine.shutdown()
