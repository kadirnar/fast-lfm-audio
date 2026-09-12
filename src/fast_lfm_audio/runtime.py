"""CUDA graphs and selective compiler fusion for the Transformers audio model.

One optimized model serves one request at a time. Weights must stay on the same
CUDA device and dtype after optimization. Prefill uses the original dynamic
cache. The FP16/BF16 path uses a preallocated static decode cache; strict FP32
retains the original dynamic cache and captures its exact shapes.
"""

import copy
import gc
import os
import types
import weakref
from collections import OrderedDict

import torch
from torch.nn.attention.varlen import varlen_attn
from transformers import DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.lfm2.modeling_lfm2 import (
    apply_rotary_pos_emb,
    create_causal_mask,
    create_recurrent_attention_mask,
)


class GraphedCall:
    def __init__(self, function, example):
        from cuda.bindings import runtime as cuda_runtime

        # Graphs keep device addresses, not Python references to closure tensors.
        self.function = function
        self.input = example.clone()
        # PyTorch's stream pool cycles through a fixed set of CUDA streams.
        # On this runtime graph.reset() clears cuBLAS workspaces for its stream.
        # Each live graph therefore needs its own stream: evicting one must not
        # free another graph's captured cuBLAS workspace.
        with torch.cuda.device(example.device):
            status, raw_stream = cuda_runtime.cudaStreamCreateWithFlags(cuda_runtime.cudaStreamNonBlocking)
            if status != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA stream creation failed: {status}")
        stream = torch.cuda.ExternalStream(int(raw_stream), device=example.device)
        self.graph = torch.cuda.CUDAGraph()
        self._finalizer = weakref.finalize(
            self, _release_graph, self.graph, cuda_runtime, raw_stream, example.device
        )
        stream.wait_stream(torch.cuda.current_stream(example.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                function(self.input)
        torch.cuda.current_stream(example.device).wait_stream(stream)
        # Collecting an older cyclic graph during capture can invoke CUDA graph
        # destruction, which CUDA forbids inside the capture region.
        collect = gc.isenabled()
        gc.disable()
        try:
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = function(self.input)
        finally:
            if collect:
                gc.enable()

    def __call__(self, value):
        self.input.copy_(value)
        self.graph.replay()
        return self.output

    def close(self):
        self.function = None
        self._finalizer()


def _release_graph(graph, cuda_runtime, stream, device):
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        graph.reset()
        (status,) = cuda_runtime.cudaStreamDestroy(stream)
        if status != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA stream destruction failed: {status}")


class GraphedAudioEncoder:
    """Replay the original encoder at each input shape; retain two recent shapes."""

    def __init__(self, model):
        self.model = model
        self.original = model.get_audio_features
        self.graphs = OrderedDict()
        model.get_audio_features = self.forward

    def forward(self, input_features, input_features_attention_mask=None):
        mask = input_features_attention_mask
        if (
            self.model.training
            or self.model.conformer.training
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or not input_features.is_cuda
            or (mask is not None and not mask.is_cuda)
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.original(input_features, mask)
        key = tuple(
            None if value is None else (value.shape, value.stride(), value.dtype, value.device)
            for value in (input_features, mask)
        )
        if key not in self.graphs:
            if len(self.graphs) >= 2:
                _, (graph, _) = self.graphs.popitem(last=False)
                graph.close()
            static_mask = None if mask is None else mask.clone()
            graph = GraphedCall(lambda value: self.original(value, static_mask), input_features)
            self.graphs[key] = (graph, static_mask)
        self.graphs.move_to_end(key)
        graph, static_mask = self.graphs[key]
        if mask is not None:
            static_mask.copy_(mask)
        audio, audio_mask = graph(input_features)
        return audio.clone(), audio_mask.clone()

    def close(self):
        self.model.get_audio_features = self.original
        for graph, _ in self.graphs.values():
            graph.close()
        self.graphs.clear()


class GraphedPrefill:
    """Capture the original prompt computation with a fresh dynamic cache."""

    def __init__(self, model, max_cache_len):
        self.model = model
        self.original = model.forward
        self.max_cache_len = max_cache_len
        self.graphs = OrderedDict()

    def __call__(self, *args, **kwargs):
        value = kwargs.get("inputs_embeds")
        if (
            args
            or value is None
            or value.ndim != 3
            or value.shape[0] != 1
            or not 1 <= value.shape[1] <= self.max_cache_len
            or not value.is_cuda
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or (
                value.dtype == torch.float32
                and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32)
            )
            or self.model.training
            or getattr(self.model.config, "output_hidden_states", False)
            or getattr(self.model.config, "output_attentions", False)
            or not getattr(self.model.config, "return_dict", True)
            or kwargs.get("past_key_values") is not None
            or kwargs.get("attention_mask") is not None
            or kwargs.get("use_cache") is not True
            or kwargs.get("return_dict", True) is not True
            or set(kwargs)
            - {"inputs_embeds", "past_key_values", "attention_mask", "use_cache", "return_dict"}
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.original(*args, **kwargs)
        key = (value.shape, value.stride(), value.dtype, value.device)
        if key not in self.graphs:
            if len(self.graphs) >= 2:
                self.graphs.popitem(last=False)[1].close()
            options = {
                "config": self.model.config,
                "inputs_embeds": value,
                "attention_mask": None,
                "past_key_values": DynamicCache(config=self.model.config),
                "position_ids": torch.arange(value.shape[1], device=value.device).unsqueeze(0),
            }
            # Freeze the eager mask choice before CUDA capture changes is_tracing().
            masks = {
                "full_attention": create_causal_mask(**options),
                "conv": create_recurrent_attention_mask(**options),
            }
            self.graphs[key] = GraphedCall(
                lambda hidden: self.original(
                    inputs_embeds=hidden, attention_mask=masks, use_cache=True, return_dict=True
                ),
                value,
            )
        self.graphs.move_to_end(key)
        output = self.graphs[key](value)
        return BaseModelOutputWithPast(
            last_hidden_state=output.last_hidden_state.clone(),
            past_key_values=copy.deepcopy(output.past_key_values),
        )

    def clear(self):
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()


@torch.library.custom_op("fast_lfm_audio::mean_last_dim", mutates_args=())
def _mean_last_dim(value: torch.Tensor) -> torch.Tensor:
    # Keep ATen's reduction order: a reassociated FP32 mean can change greedy audio codes.
    return value.mean(-1, keepdim=True)


@_mean_last_dim.register_fake
def _mean_last_dim_fake(value):
    return value.new_empty((*value.shape[:-1], 1))


def _depth_backend(graph, inputs):
    for node in graph.graph.nodes:
        if node.op == "call_method" and node.target == "mean":
            if node.args[1:] != (-1,) or node.kwargs != {"keepdim": True}:
                raise ValueError("Unsupported Depthformer reduction signature.")
            node.op = "call_function"
            node.target = torch.ops.fast_lfm_audio.mean_last_dim.default
            node.args, node.kwargs = (node.args[0],), {}
    graph.recompile()
    return torch._inductor.compile(
        graph,
        inputs,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True, "compile_threads": 4},
    )


def depth_graph(function, example):
    """Fuse greedy depth operations, preserving reductions and precision casts."""
    if os.environ.get("FAST_LFM_COMPILE_DEPTH", "1") != "0":
        # Build lazy complex rotary buffers with the original operations, outside the compiler.
        function(example)
        function = torch.compile(function, backend=_depth_backend, fullgraph=True, dynamic=False)
    return GraphedCall(function, example)


class GraphedBackbone:
    def __init__(self, backbone, max_cache_len):
        self.backbone = backbone
        self.original = backbone.forward
        self.prefill = GraphedPrefill(backbone, max_cache_len)
        self.max_cache_len = max_cache_len
        self.cache = None
        self.calls = {}
        self.length = 0
        self.attention_originals = []
        self.attention_max_k = max_cache_len

    def _install_attention(self, example):
        self.cu_query = torch.tensor([0, 1], device=example.device, dtype=torch.int32)
        self.cu_key = torch.tensor([0, self.max_cache_len], device=example.device, dtype=torch.int32)
        for layer in self.backbone.layers:
            if not layer.is_attention_layer:
                continue
            attention = layer.self_attn
            original = attention.forward
            self.attention_originals.append((attention, original))

            def forward(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                _original=original,
                **kwargs,
            ):
                if past_key_values is not self.cache or hidden_states.shape[:2] != (1, 1):
                    return _original(
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values=past_key_values,
                        **kwargs,
                    )
                shape = (*hidden_states.shape[:-1], -1, module.head_dim)
                query = module.q_layernorm(module.q_proj(hidden_states).view(shape)).transpose(1, 2)
                key = module.k_layernorm(module.k_proj(hidden_states).view(shape)).transpose(1, 2)
                value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
                query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
                key, value = self.cache.update(key, value, module.layer_idx)
                length = self.cache.layers[module.layer_idx].cumulative_length.reshape(1).to(torch.int32)
                output = varlen_attn(
                    query[0].transpose(0, 1),
                    key[0].transpose(0, 1),
                    value[0].transpose(0, 1),
                    self.cu_query,
                    self.cu_key,
                    1,
                    self.attention_max_k,
                    scale=module.scaling,
                    enable_gqa=True,
                    seqused_k=length,
                )
                return module.out_proj(output.reshape(1, 1, -1)), None

            attention.forward = types.MethodType(forward, attention)

    def _initialize(self, example):
        self.cache = StaticCache(config=self.backbone.config, max_cache_len=self.max_cache_len)
        self._install_attention(example)
        self.original(inputs_embeds=example.expand(1, 4, -1), past_key_values=self.cache, use_cache=True)

    def _capture(self, example, bucket):
        self.attention_max_k = bucket
        # Warmup and capture advance recurrent/KV state, so restore it before replay.
        states = []
        for layer in self.cache.layers:
            if hasattr(layer, "keys"):
                states.extend((layer.keys, layer.values, layer.cumulative_length))
            else:
                states.extend(state for state in layer.conv_states.values() if state is not None)
        saved = [state.clone() for state in states]
        for layer in self.cache.layers:
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length.zero_()

        def step(embedding):
            position = self.cache.get_seq_length().clone().reshape(1, 1)
            return self.original(
                inputs_embeds=embedding,
                position_ids=position,
                attention_mask={"full_attention": None, "conv": None},
                past_key_values=self.cache,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state

        self.calls[bucket] = GraphedCall(step, example)
        for state, original in zip(states, saved, strict=True):
            state.copy_(original)

    def _copy_prefill(self, source):
        self.length = int(source.get_seq_length())
        if self.length >= self.max_cache_len:
            raise ValueError(f"Prompt exceeds graph cache capacity ({self.max_cache_len} tokens).")
        for target, origin in zip(self.cache.layers, source.layers, strict=True):
            if hasattr(target, "keys"):
                target.keys.zero_()
                target.values.zero_()
                target.keys[:, :, : self.length].copy_(origin.keys)
                target.values[:, :, : self.length].copy_(origin.values)
                target.cumulative_length.fill_(self.length)
            else:
                for state_idx, state in origin.conv_states.items():
                    if state is not None:
                        target.conv_states[state_idx].copy_(state)
                        target.has_previous_state[state_idx] = True

    def forward(self, *args, **kwargs):
        embedding = kwargs.get("inputs_embeds")
        past = kwargs.get("past_key_values")
        if past is None:
            return self.prefill(*args, **kwargs)
        if args or embedding is None or embedding.shape[:2] != (1, 1):
            if past is self.cache:
                raise ValueError("Graph cache supports batch-one, single-token decoding only.")
            return self.original(*args, **kwargs)
        if self.cache is None:
            self._initialize(embedding)
        if past is not self.cache:
            self._copy_prefill(past)
        if self.length >= self.max_cache_len:
            raise ValueError(f"Generation exceeds graph cache capacity ({self.max_cache_len} tokens).")
        self.length += 1
        # Flash Attention chooses split-KV reductions from max_k. Small context
        # buckets preserve that choice instead of treating every step as max length.
        bucket = min(((self.length + 255) // 256) * 256, self.max_cache_len)
        if bucket not in self.calls:
            self._capture(embedding, bucket)
        hidden = self.calls[bucket](embedding)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=self.cache)


class Optimization:
    def __init__(self, model, *, backbone=True, depth=True, max_cache_len=2048):
        self.model = model
        self.original_sample = model._sample_audio_frame
        self.original_forward = model.model.lfm.forward
        self.depth_graphs = {}
        self.encoder = GraphedAudioEncoder(model.model)
        self.closed = False
        self.backbone = GraphedBackbone(model.model.lfm, max_cache_len) if backbone else None
        if self.backbone is not None:
            model.model.lfm.forward = self.backbone.forward
        if depth:
            model._sample_audio_frame = self.sample

    def sample(self, hidden_state, temperature, top_k):
        greedy = temperature is None or temperature <= 0 or top_k == 1
        key = (None, 1) if greedy else (temperature, top_k)
        if key not in self.depth_graphs:
            graph = depth_graph if greedy else GraphedCall
            self.depth_graphs[key] = graph(
                lambda hidden: self.original_sample(hidden, temperature=key[0], top_k=key[1]), hidden_state
            )
        # Generation retains every frame; graph outputs alias the next replay.
        return self.depth_graphs[key](hidden_state).clone()

    def close(self):
        if self.closed:
            return
        self.model._sample_audio_frame = self.original_sample
        self.model.model.lfm.forward = self.original_forward
        self.encoder.close()
        for graph in self.depth_graphs.values():
            graph.close()
        self.depth_graphs.clear()
        if self.backbone is not None:
            self.backbone.prefill.clear()
            for module, original in self.backbone.attention_originals:
                module.forward = original
            for graph in self.backbone.calls.values():
                graph.close()
            self.backbone.calls.clear()
        del self.model._fast_lfm_optimization
        self.closed = True


def optimize(model, *, backbone=True, depth=True, max_cache_len=2048, strict=None):
    """Optimize in place; FP32 defaults to strict fusion with the original gradient path.

    Reduced-precision graphs support FP16 and BF16 inference.
    Strict mode preserves the original backbone attention and dynamic cache shapes.
    """
    if model.training:
        raise ValueError("Call model.eval() before enabling inference graphs.")
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("CUDA graphs require the model on a CUDA device.")
    if max_cache_len < 16:
        raise ValueError("max_cache_len must be at least 16.")
    if hasattr(model, "_fast_lfm_optimization"):
        raise ValueError("This model is already optimized.")
    if strict is None:
        strict = next(model.parameters()).dtype == torch.float32
    if strict:
        from .strict import StrictOptimization

        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in model.parameters()
        ):
            raise ValueError("Strict mode requires FP32 model parameters; load with dtype=torch.float32.")
        handle = StrictOptimization(model, backbone=backbone, depth=depth, max_cache_len=max_cache_len)
    else:
        if next(model.parameters()).dtype == torch.float32 and backbone:
            raise ValueError(
                "FP32 backbone decoding requires strict=True; the legacy attention kernel uses BF16/FP16."
            )
        handle = Optimization(model, backbone=backbone, depth=depth, max_cache_len=max_cache_len)
    model._fast_lfm_optimization = handle
    return handle
