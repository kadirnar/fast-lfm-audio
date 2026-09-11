import pytest
import torch
from transformers import Lfm2AudioConfig, Lfm2AudioForConditionalGeneration

from fast_lfm_audio import optimize
from fast_lfm_audio.pipeline import decodable_codes
from fast_lfm_audio.runtime import GraphedBackbone


@pytest.fixture
def model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(123)
    torch.set_num_threads(4)
    config = Lfm2AudioConfig(
        codebooks=8,
        audio_vocab_size=33,
        audio_eos_token_id=32,
        preprocessor={"features": 8},
        encoder={"feat_in": 8, "n_layers": 1, "d_model": 16, "n_heads": 4, "subsampling_conv_channels": 8},
        lfm={
            "vocab_size": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention", "conv"],
        },
        depthformer={
            "layers": 2,
            "dim": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
        },
    )
    return Lfm2AudioForConditionalGeneration(config).to("cuda", torch.bfloat16).eval()


@torch.inference_mode()
def test_depth_graph_preserves_frames_and_owns_outputs(model):
    inputs = [torch.randn(64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    expected = [model._sample_audio_frame(hidden, None, 1) for hidden in inputs]
    handle = optimize(model, backbone=False)
    actual = [model._sample_audio_frame(hidden, None, 1) for hidden in inputs]
    for want, got in zip(expected, actual, strict=True):
        assert torch.equal(want, got)
    assert len({frame.data_ptr() for frame in actual}) == len(actual)
    handle.close()
    optimize(model, backbone=False).close()


@torch.inference_mode()
def test_backbone_resets_between_different_prompts(model):
    backbone = model.model.lfm
    original = backbone.forward
    engine = GraphedBackbone(backbone, 768)
    for prompt_length in (3, 31, 7, 250, 508):
        prompt = torch.randn(1, prompt_length, 64, device="cuda", dtype=torch.bfloat16)
        reference_cache = original(inputs_embeds=prompt, use_cache=True).past_key_values
        actual_cache = original(inputs_embeds=prompt, use_cache=True).past_key_values
        for _ in range(12):
            embedding = torch.randn(1, 1, 64, device="cuda", dtype=torch.bfloat16)
            expected = original(inputs_embeds=embedding, past_key_values=reference_cache, use_cache=True)
            actual = engine.forward(inputs_embeds=embedding, past_key_values=actual_cache, use_cache=True)
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
            reference_cache, actual_cache = expected.past_key_values, actual.past_key_values


@torch.inference_mode()
def test_cache_capacity_checked_before_device_write(model):
    backbone = model.model.lfm
    engine = GraphedBackbone(backbone, 16)
    prompt = torch.randn(1, 15, 64, device="cuda", dtype=torch.bfloat16)
    past = backbone(inputs_embeds=prompt, use_cache=True).past_key_values
    embedding = prompt[:, :1].contiguous()
    result = engine.forward(inputs_embeds=embedding, past_key_values=past, use_cache=True)
    with pytest.raises(ValueError, match="exceeds graph cache capacity"):
        engine.forward(inputs_embeds=embedding, past_key_values=result.past_key_values, use_cache=True)
    torch.cuda.synchronize()


@torch.inference_mode()
def test_stochastic_graph_replay_advances_rng(model):
    optimize(model, backbone=False)
    hidden = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    frames = [model._sample_audio_frame(hidden, 1.0, 4).cpu() for _ in range(5)]
    assert all(bool(((frame >= 0) & (frame < 33)).all()) for frame in frames)
    assert any(not torch.equal(frames[0], frame) for frame in frames[1:])


def test_audio_eos_is_removed_without_discarding_last_real_frame():
    codes = torch.arange(24).reshape(1, 8, 3)
    assert torch.equal(decodable_codes(codes), codes)
    eos = torch.full((1, 8, 1), 2048)
    assert torch.equal(decodable_codes(torch.cat((codes, eos), -1)), codes)
    assert decodable_codes(eos).shape == (1, 8, 0)
    with pytest.raises(ValueError, match="Audio codes"):
        decodable_codes(torch.cat((eos, codes), -1))
