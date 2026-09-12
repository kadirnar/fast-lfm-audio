"""SGLang's native LFM2 cache and kernels with an out-of-graph audio sampler."""

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import torch

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
            incremental_streaming_output=False,
        )
        self.embedding_directory = tempfile.TemporaryDirectory(prefix="fast-lfm-audio-", dir="/dev/shm")
        self.embedding_path = str(Path(self.embedding_directory.name) / "prompt.bin")
        self.embedding_buffer = None
        self.max_cache_len = max_cache_len

    def generate(self, embeddings, mode, max_new_tokens, *, on_event=None):
        from sglang.srt.managers.io_struct import GenerateReqInput

        frames, width = embeddings.shape
        if self.embedding_buffer is None:
            self.embedding_buffer = torch.from_file(
                self.embedding_path, shared=True, size=self.max_cache_len * width, dtype=torch.bfloat16
            ).reshape(self.max_cache_len, width)
        # Transfer the original BF16 bits once. Nested Python float lists make
        # long prompts expensive to serialize and reconstruct in the scheduler.
        self.embedding_buffer[:frames].copy_(embeddings)
        self.engine.collective_rpc(
            "reset_lfm_audio", mode=mode, embedding_path=self.embedding_path, frames=frames, width=width
        )
        request = GenerateReqInput(
            rid=str(uuid4()),
            input_ids=[0] * frames,
            sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
            stream=on_event is not None,
        )
        manager = self.engine.tokenizer_manager
        generator = manager.generate_request(request, None)
        events = []
        try:
            while True:
                try:
                    output = self.engine.loop.run_until_complete(generator.__anext__())
                except StopAsyncIteration:
                    break
                # SGLang returns cumulative custom metadata in this configuration.
                pending = output["meta_info"]["lfm_audio_events"][len(events) :]
                events.extend(pending)
                if on_event is not None:
                    for event in pending:
                        on_event(event)
        except BaseException:
            manager.abort_request(request.rid)

            async def drain():
                async for _ in generator:
                    pass

            self.engine.loop.run_until_complete(drain())
            raise
        finally:
            self.engine.loop.run_until_complete(generator.aclose())
        return events

    def close(self):
        try:
            self.engine.shutdown()
        finally:
            self.embedding_buffer = None
            self.embedding_directory.cleanup()
