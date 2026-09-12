"""Strict FP32 inference with reference-ordered kernels and unchanged autograd graphs."""

import math
import os
import struct
from collections import OrderedDict
from contextlib import contextmanager

import torch
from transformers.models.lfm2.modeling_lfm2 import Lfm2MLP, Lfm2RMSNorm
from transformers.models.lfm2_audio.modeling_lfm2_audio import (
    Lfm2AudioDepthAttention,
    Lfm2AudioDepthAttentionCore,
    Lfm2AudioDepthMLP,
    Lfm2AudioRMSNorm,
    _apply_depth_rotary,
)
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiTransformerLayer
from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderBlock

from .kernels import _eligible, rms_norm, scaled_add, silu_mul
from .packed import PackedLinear, supported_runtime
from .strict_attention import depth_attention
from .strict_generation import generate as generate_with_projection
from .strict_generation import matches_reference


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _single_input(args, kwargs, name):
    if len(args) == 1 and not kwargs:
        return args[0]
    if not args and len(kwargs) == 1:
        return kwargs.get(name)
    return None


def _extra_padding(length, kernel, stride, padding):
    # Mirror the reference's integer -> default-float division and +1 rounding.
    # Padding depends only on shapes; evaluating it on the host removes GPU .item()
    # synchronization without changing a convolution input or reduction.
    cast = _f32 if torch.get_default_dtype() == torch.float32 else float
    frames = math.ceil(cast(cast(cast(length - kernel + padding) / cast(stride)) + 1)) - 1
    return frames * stride + kernel - padding - length


@contextmanager
def fp32_precision():
    """Use IEEE FP32 for this single-request runtime, restoring caller settings."""
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise ValueError("Strict FP32 requires TORCH_ALLOW_TF32_CUBLAS_OVERRIDE to be unset.")
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class ExactModules:
    """Install only known equivalent FP32 formulas; keep the original gradient path."""

    def __init__(self, model, *, backbone=True, depth=True, graph_backbone=False):
        self.originals = []
        self.graphs = {}
        self.packed = []
        self.counts = {"rms_norm": 0, "silu_mul": 0, "conformer": 0, "mimi": 0, "mimi_conv": 0}
        known_runtime = next(model.parameters()).is_cuda and supported_runtime(
            next(model.parameters()).device
        )
        pack = known_runtime and os.environ.get("FAST_LFM_STRICT_PACKED_LINEAR", "1") == "1"
        for name, module in model.named_modules():
            if (not backbone and name.startswith("model.lfm.")) or (
                not depth and name.startswith(("model.depthformer.", "model.depth_embeddings."))
            ):
                continue
            kind = type(module)
            if kind in (Lfm2RMSNorm, Lfm2AudioRMSNorm):
                self._norm(module, kind is Lfm2RMSNorm)
            elif kind in (Lfm2MLP, Lfm2AudioDepthMLP):
                self._mlp(module, "x" if kind is Lfm2MLP else "hidden_states")
            elif kind is ParakeetEncoderBlock:
                self._conformer(module)
            elif kind is MimiTransformerLayer:
                self._mimi(module)
            elif kind is MimiConv1d:
                self._mimi_conv(module)
            elif (
                known_runtime
                and kind is Lfm2AudioDepthAttentionCore
                and os.environ.get("FAST_LFM_STRICT_DEPTH_ATTENTION", "1") == "1"
            ):
                self._depth_attention(module)
            elif pack and kind is torch.nn.Linear:
                packed = PackedLinear.create(module)
                if packed is not None:
                    self.packed.append(packed)
                    self._install(module, packed)
            if graph_backbone and name.startswith("model.lfm.") and kind in (Lfm2RMSNorm, Lfm2MLP):
                self._graph_module(module, "x" if kind is Lfm2MLP else "hidden_states")

    def _install(self, module, function):
        # Preserve whether the instance had its own forward attribute.
        self.originals.append((module, module.__dict__.get("forward")))
        module.forward = function

    def _norm(self, module, weight_first):
        original = module.forward

        def forward(hidden_states):
            if module.training or not _eligible(hidden_states, module.weight):
                return original(hidden_states)
            return rms_norm(hidden_states, module.weight, module.variance_epsilon, weight_first=weight_first)

        self._install(module, forward)
        self.counts["rms_norm"] += 1

    def _mlp(self, module, input_name):
        original = module.forward

        def forward(*args, **kwargs):
            value = _single_input(args, kwargs, input_name)
            if value is None or module.training or not _eligible(value, module.w1.weight):
                return original(*args, **kwargs)
            return module.w2(silu_mul(module.w1(value), module.w3(value)))

        self._install(module, forward)
        self.counts["silu_mul"] += 1

    def _conformer(self, module):
        original = module.forward

        def forward(hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
            if module.training or not _eligible(hidden_states):
                return original(hidden_states, attention_mask, position_embeddings, **kwargs)
            value = module.feed_forward1(module.norm_feed_forward1(hidden_states))
            hidden_states = scaled_add(hidden_states, value)
            attention, _ = module.self_attn(
                hidden_states=module.norm_self_att(hidden_states),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = hidden_states + attention
            hidden_states = hidden_states + module.conv(
                module.norm_conv(hidden_states), attention_mask=attention_mask
            )
            value = module.feed_forward2(module.norm_feed_forward2(hidden_states))
            return module.norm_out(scaled_add(hidden_states, value))

        self._install(module, forward)
        self.counts["conformer"] += 1

    def _depth_attention(self, module):
        original = module.forward

        def forward(query_states, key_states, value_states, frequencies, past_key_value=None):
            length = 1 if past_key_value is None else past_key_value[0].shape[1] + 1
            if (
                module.training
                or module.num_attention_heads != 32
                or module.num_key_value_heads != 8
                or module.head_dim != 32
                or query_states.shape != (1, 1, 1024)
                or key_states.shape != (1, 1, 256)
                or value_states.shape != (1, 1, 256)
                or not 1 <= length <= 8
                or not _eligible(query_states, key_states, value_states)
                or frequencies.shape != (1, 16)
                or frequencies.dtype != torch.complex64
                or frequencies.device != query_states.device
                or (
                    past_key_value is not None
                    and (
                        len(past_key_value) != 2
                        or past_key_value[0].shape != (1, length - 1, 8, 32)
                        or past_key_value[1].shape != (1, length - 1, 8, 32)
                        or not _eligible(*past_key_value)
                        or past_key_value[0].device != query_states.device
                    )
                )
            ):
                return original(query_states, key_states, value_states, frequencies, past_key_value)
            query = module.q_layernorm(query_states.view(1, 1, 32, 32))
            key = module.k_layernorm(key_states.view(1, 1, 8, 32))
            value = value_states.view(1, 1, 8, 32)
            query, key = _apply_depth_rotary(query, key, frequencies)
            if past_key_value is not None:
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)
            hidden = depth_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
            return hidden.transpose(1, 2).reshape(1, 1, 1024), (key, value)

        self._install(module, forward)

    def _mimi(self, module):
        original = module.forward

        def forward(
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=None,
            **kwargs,
        ):
            if module.training or not _eligible(hidden_states):
                return original(
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    position_embeddings,
                    **kwargs,
                )
            value, attention = module.self_attn(
                hidden_states=module.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = scaled_add(hidden_states, value, module.self_attn_layer_scale.scale)
            value = module.mlp(module.post_attention_layernorm(hidden_states))
            hidden_states = scaled_add(hidden_states, value, module.mlp_layer_scale.scale)
            return (hidden_states, attention) if output_attentions else (hidden_states,)

        self._install(module, forward)
        self.counts["mimi"] += 1

    def _mimi_conv(self, module):
        original = module.forward
        stride = int(module.stride)
        kernel = int(module.kernel_size)
        padding = int(module.padding_total)
        right = padding // 2
        left = padding - right

        def forward(hidden_states, padding_cache=None):
            if (
                module.training
                or torch.is_grad_enabled()
                or padding_cache is not None
                or torch.get_default_dtype() not in (torch.float32, torch.float64)
            ):
                return original(hidden_states, padding_cache)
            extra = _extra_padding(hidden_states.shape[-1], kernel, stride, padding)
            pads = (padding, extra) if module.causal else (left, right + extra)
            return module.conv(module._pad1d(hidden_states, pads, mode=module.pad_mode))

        self._install(module, forward)
        self.counts["mimi_conv"] += 1

    def _graph_module(self, module, input_name):
        from .runtime import GraphedCall

        original = module.forward

        def forward(*args, **kwargs):
            value = _single_input(args, kwargs, input_name)
            if (
                value is None
                or value.ndim < 3
                or value.shape[:2] != (1, 1)
                or module.training
                or not _eligible(value)
                or torch.cuda.is_current_stream_capturing()
            ):
                return original(*args, **kwargs)
            key = (id(module), value.shape, value.dtype, value.device)
            if key not in self.graphs:
                self.graphs[key] = GraphedCall(original, value)
            return self.graphs[key](value).clone()

        self._install(module, forward)

    def close(self):
        for module, original in reversed(self.originals):
            if original is None:
                del module.forward
            else:
                module.forward = original
        self.originals.clear()
        self.graphs.clear()
        self.packed.clear()


class StrictOptimization:
    """Preserve reference arithmetic, cache shapes, and gradient computations."""

    def __init__(self, model, *, backbone=True, depth=True, max_cache_len=2048):
        from .runtime import GraphedCall

        self.model = model
        self.depth_graphs = {}
        self.encoder_graphs = OrderedDict()
        self.original_sample = model._sample_audio_frame
        self.original_audio_features = model.model.get_audio_features
        self.original_generate = model.generate
        self.original_forward = model.forward
        # A lazy buffer created in inference_mode cannot later be saved for backward.
        # Initialize it with the original arithmetic and normal tensor ownership.
        with torch.inference_mode(False), torch.no_grad():
            for module in model.modules():
                if type(module) is Lfm2AudioDepthAttention:
                    if module.frequencies is not None and module.frequencies.is_inference():
                        module.frequencies = None
                    module.get_frequencies(next(module.parameters()).device)
        self.modules = ExactModules(
            model,
            backbone=backbone,
            depth=depth,
            graph_backbone=os.environ.get("FAST_LFM_STRICT_MODULE_GRAPHS", "1") == "1",
        )
        self.backbone_graph = None
        self._in_generate = False
        self.text_projection = None
        if (
            os.environ.get("FAST_LFM_STRICT_PACKED_LINEAR", "1") == "1"
            and os.environ.get("FAST_LFM_STRICT_TEXT_PROJECTION", "1") == "1"
            and supported_runtime(next(model.parameters()).device)
            and matches_reference(self.original_generate)
        ):
            self.text_projection = PackedLinear.create(model.get_output_embeddings(), projection=True)
            if self.text_projection is not None:
                self.modules.packed.append(self.text_projection)
        self._packed_versions = self._weight_versions()
        if backbone and os.environ.get("FAST_LFM_STRICT_BACKBONE_GRAPHS", "1") == "1":
            from .strict_backbone import ExactBackbone

            self.backbone_graph = ExactBackbone(
                model.model.lfm, refresh=self._refresh_weights, max_cache_len=max_cache_len
            )
        self.closed = False

        def sample(hidden_state, temperature, top_k):
            # Stochastic capture consumes random numbers during warmup. Keep its original RNG sequence.
            greedy = temperature is None or temperature <= 0 or top_k == 1
            if model.training or not _eligible(hidden_state) or not greedy:
                return self.original_sample(hidden_state, temperature, top_k)
            self._refresh_weights()
            key = (hidden_state.shape, hidden_state.dtype, hidden_state.device)
            if key not in self.depth_graphs:
                self.depth_graphs[key] = GraphedCall(
                    lambda hidden: self.original_sample(hidden, None, 1), hidden_state
                )
            return self.depth_graphs[key](hidden_state).clone()

        def generate(*args, **kwargs):
            with fp32_precision():
                self._refresh_weights()
                previous = self._in_generate
                self._in_generate = True
                try:
                    if (
                        self.text_projection is not None
                        and self.text_projection.module is model.get_output_embeddings()
                    ):
                        return generate_with_projection(model, self.text_projection, *args, **kwargs)
                    return self.original_generate(*args, **kwargs)
                finally:
                    self._in_generate = previous

        def forward(*args, **kwargs):
            with fp32_precision():
                self._refresh_weights()
                return self.original_forward(*args, **kwargs)

        def audio_features(input_features, input_features_attention_mask=None):
            mask = input_features_attention_mask
            # The processor returns a dense transpose of mel bins and time.
            eligible = _eligible(input_features) or (
                input_features.ndim >= 2 and _eligible(input_features.transpose(-1, -2))
            )
            if (
                model.training
                or model.model.conformer.training
                or not eligible
                or (mask is not None and not mask.is_cuda)
            ):
                return self.original_audio_features(input_features, mask)
            key = (
                input_features.shape,
                input_features.stride(),
                input_features.dtype,
                input_features.device,
                None if mask is None else (mask.shape, mask.stride(), mask.dtype, mask.device),
            )
            if key not in self.encoder_graphs:
                if len(self.encoder_graphs) >= 2:
                    self.encoder_graphs.popitem(last=False)
                static_mask = None if mask is None else mask.clone()
                graph = GraphedCall(
                    lambda value: self.original_audio_features(value, static_mask), input_features
                )
                self.encoder_graphs[key] = (graph, static_mask)
            self.encoder_graphs.move_to_end(key)
            graph, static_mask = self.encoder_graphs[key]
            if mask is not None:
                static_mask.copy_(mask)
            audio, audio_mask = graph(input_features)
            return audio.clone(), audio_mask.clone()

        if depth:
            model._sample_audio_frame = sample
        model.generate = generate
        model.forward = forward
        if os.environ.get("FAST_LFM_STRICT_ENCODER_GRAPHS", "1") == "1":
            model.model.get_audio_features = audio_features

    def _weight_versions(self):
        return tuple(
            (
                id(packed.module.weight),
                None if packed.module.weight.is_inference() else packed.module.weight._version,
            )
            for packed in self.modules.packed
        )

    def _refresh_weights(self):
        # This runtime serves one request at a time. Check once at generation
        # entry; weights stay fixed throughout its autoregressive loop.
        if self._in_generate:
            return
        versions = self._weight_versions()
        if versions != self._packed_versions:
            self.modules.graphs.clear()
            self.depth_graphs.clear()
            if self.backbone_graph is not None:
                self.backbone_graph.clear()
            self._packed_versions = versions

    def close(self):
        if self.closed:
            return
        self.model._sample_audio_frame = self.original_sample
        self.model.generate = self.original_generate
        self.model.forward = self.original_forward
        self.model.model.get_audio_features = self.original_audio_features
        if self.backbone_graph is not None:
            self.backbone_graph.close()
        self.modules.close()
        self.text_projection = None
        self.depth_graphs.clear()
        self.encoder_graphs.clear()
        del self.model._fast_lfm_optimization
        self.closed = True
