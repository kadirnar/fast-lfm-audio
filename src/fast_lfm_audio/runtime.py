"""CUDA graphs around the original Transformers model operations.

One optimized model serves one request at a time. Weights must stay on the same
CUDA device and dtype after optimization. Prefill uses the original dynamic
cache; only single-token decoding uses a preallocated static cache.
"""

import types

import torch
from torch.nn.attention.varlen import varlen_attn
from transformers import StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb


class GraphedCall:
    def __init__(self, function, example):
        # Graphs keep device addresses, not Python references to closure tensors.
        self.function = function
        self.input = example.clone()
        stream = torch.cuda.Stream(device=example.device)
        stream.wait_stream(torch.cuda.current_stream(example.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                function(self.input)
        torch.cuda.current_stream(example.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = function(self.input)

    def __call__(self, value):
        self.input.copy_(value)
        self.graph.replay()
        return self.output


class GraphedBackbone:
    def __init__(self, backbone, max_cache_len):
        self.backbone = backbone
        self.original = backbone.forward
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
            return self.original(*args, **kwargs)
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
        self.backbone = GraphedBackbone(model.model.lfm, max_cache_len) if backbone else None
        if self.backbone is not None:
            model.model.lfm.forward = self.backbone.forward
        if depth:
            model._sample_audio_frame = self.sample

    def sample(self, hidden_state, temperature, top_k):
        greedy = temperature is None or temperature <= 0 or top_k == 1
        key = (None, 1) if greedy else (temperature, top_k)
        if key not in self.depth_graphs:
            self.depth_graphs[key] = GraphedCall(
                lambda hidden: self.original_sample(hidden, temperature=key[0], top_k=key[1]), hidden_state
            )
        # Generation retains every frame; graph outputs alias the next replay.
        return self.depth_graphs[key](hidden_state).clone()

    def close(self):
        self.model._sample_audio_frame = self.original_sample
        self.model.model.lfm.forward = self.original_forward
        if self.backbone is not None:
            for module, original in self.backbone.attention_originals:
                module.forward = original
        del self.model._fast_lfm_optimization


def optimize(model, *, backbone=True, depth=True, max_cache_len=2048):
    """Optimize a loaded, evaluated PR-48249 model in place; return a restore handle."""
    if model.training:
        raise ValueError("Call model.eval() before enabling inference graphs.")
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("CUDA graphs require the model on a CUDA device.")
    if max_cache_len < 16:
        raise ValueError("max_cache_len must be at least 16.")
    if hasattr(model, "_fast_lfm_optimization"):
        raise ValueError("This model is already optimized.")
    handle = Optimization(model, backbone=backbone, depth=depth, max_cache_len=max_cache_len)
    model._fast_lfm_optimization = handle
    return handle
