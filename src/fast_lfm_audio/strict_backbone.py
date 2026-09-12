"""CUDA graphs of the original single-token LFM2 computation at exact cache lengths."""

import copy
import weakref
from collections import OrderedDict

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.lfm2.modeling_lfm2 import create_causal_mask, create_recurrent_attention_mask

from .kernels import _eligible


class _Step:
    def __init__(self, model, function, value, cache, options):
        from .runtime import GraphedCall

        self.cache = copy.deepcopy(cache)
        self.states = []
        for layer in self.cache.layers:
            if type(layer) is DynamicLayer:
                self.states.append((layer.keys, layer.values))
            else:
                self.states.append((layer.conv_states[0].clone(),))
        # Transformers' is_tracing() includes CUDA capture. Compute the eager mask
        # choice first so capture cannot change GQA or the attention implementation.
        positions = (torch.arange(1, device=value.device) + cache.get_seq_length()).unsqueeze(0)
        mask_options = {
            "config": model.config,
            "inputs_embeds": value,
            "attention_mask": None,
            "past_key_values": cache,
            "position_ids": positions,
        }
        options = {
            **options,
            "attention_mask": {
                "full_attention": create_causal_mask(**mask_options),
                "conv": create_recurrent_attention_mask(**mask_options),
            },
        }

        def execute(hidden):
            for layer, state in zip(self.cache.layers, self.states, strict=True):
                if len(state) == 2:
                    layer.keys, layer.values = state
                else:
                    layer.conv_states[0].copy_(state[0])
            return function(inputs_embeds=hidden, past_key_values=self.cache, **options).last_hidden_state

        self.graph = GraphedCall(execute, value)

    def __call__(self, value, cache):
        for layer, state in zip(cache.layers, self.states, strict=True):
            if len(state) == 2:
                state[0].copy_(layer.keys)
                state[1].copy_(layer.values)
            else:
                state[0].copy_(layer.conv_states[0])
        hidden = self.graph(value).clone()
        for dest, source in zip(cache.layers, self.cache.layers, strict=True):
            if type(source) is DynamicLayer:
                dest.keys = source.keys.clone()
                dest.values = source.values.clone()
            else:
                dest.conv_states[0].copy_(source.conv_states[0])
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache)

    def close(self):
        # The capture closure refers to this step. Break that cycle and release
        # the CUDA graph now, so Python GC cannot destroy it during a later capture.
        self.graph.close()


class ExactBackbone:
    def __init__(self, model, *, refresh, max_cache_len=2048, capacity=64):
        self.model = model
        self.original = model.forward
        self.instance_forward = model.__dict__.get("forward")
        self.refresh = refresh
        self.max_cache_len = max_cache_len
        self.capacity = capacity
        self.graphs = OrderedDict()
        self._request_cache = None
        self._request_step = 0
        model.forward = self.forward

    def _can_graph(self, args, kwargs):
        value, cache = kwargs.get("inputs_embeds"), kwargs.get("past_key_values")
        if (
            args
            or type(cache) is not DynamicCache
            or value is None
            or value.ndim != 3
            or value.shape[:2] != (1, 1)
            or not _eligible(value)
            or self.model.training
            or getattr(self.model.config, "output_hidden_states", False)
            or getattr(self.model.config, "output_attentions", False)
            or not getattr(self.model.config, "return_dict", True)
            or torch.cuda.is_current_stream_capturing()
            or kwargs.get("attention_mask") is not None
            or kwargs.get("use_cache") is not True
            or kwargs.get("return_dict", True) is not True
            or set(kwargs)
            - {"inputs_embeds", "past_key_values", "attention_mask", "use_cache", "return_dict"}
            or cache.get_seq_length() + 1 > self.max_cache_len
            or getattr(cache, "offloading", False)
            or len(cache.layers) != len(self.model.layers)
        ):
            return False
        for layer in cache.layers:
            if type(layer) is DynamicLayer:
                if not layer.is_initialized or not _eligible(layer.keys, layer.values):
                    return False
            elif type(layer) is LinearAttentionLayer:
                if (
                    layer.record_past
                    or layer.number_of_states != 1
                    or not layer.has_previous_state[0]
                    or not layer.is_conv_states_initialized[0]
                    or not _eligible(layer.conv_states[0])
                ):
                    return False
            else:
                return False
        return True

    def forward(self, *args, **kwargs):
        if not self._can_graph(args, kwargs):
            return self.original(*args, **kwargs)
        self.refresh()
        value, cache = kwargs["inputs_embeds"], kwargs["past_key_values"]
        if self._request_cache is None or self._request_cache() is not cache:
            self._request_cache = weakref.ref(cache)
            self._request_step = 0
        self._request_step += 1
        states = tuple(
            tuple((x.shape, x.stride(), x.dtype, x.device) for x in (layer.keys, layer.values))
            if type(layer) is DynamicLayer
            else (
                layer.conv_states[0].shape,
                layer.conv_states[0].stride(),
                layer.conv_states[0].dtype,
                layer.conv_states[0].device,
            )
            for layer in cache.layers
        )
        key = (value.shape, value.stride(), value.dtype, value.device, states)
        if key not in self.graphs:
            # Long responses otherwise evict every first-audio graph, forcing
            # capture again at the start of the next request. Admit new shapes
            # only during the first capacity decode steps of each request.
            # Later misses retain the original exact computation and autograd.
            if self._request_step > self.capacity:
                return self.original(*args, **kwargs)
            if len(self.graphs) >= self.capacity:
                _, step = self.graphs.popitem(last=False)
                step.close()
            options = {k: v for k, v in kwargs.items() if k not in ("inputs_embeds", "past_key_values")}
            self.graphs[key] = _Step(self.model, self.original, value, cache, options)
        self.graphs.move_to_end(key)
        return self.graphs[key](value, cache)

    def clear(self):
        for step in self.graphs.values():
            step.close()
        self.graphs.clear()
        self._request_cache = None
        self._request_step = 0

    def close(self):
        if self.instance_forward is None:
            del self.model.forward
        else:
            self.model.forward = self.instance_forward
        self.clear()
