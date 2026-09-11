import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fast_lfm_audio import Pipeline
from fast_lfm_audio.backends.audio import AudioState, output_tensors
from fast_lfm_audio.backends.model import NativeGenerator, backbone_path, export_backbone
from fast_lfm_audio.backends.plugins import register_sglang, register_vllm


@pytest.fixture
def state():
    state = object.__new__(AudioState)
    state.config = SimpleNamespace(
        interleaved_n_text=2,
        interleaved_n_audio=2,
        eos_token_id=7,
        audio_start_token_id=128,
        audio_token_id=133,
        text_end_token_id=130,
        audio_eos_token_id=2048,
    )
    state.next_embedding = torch.zeros(4)
    state.use_audio = torch.tensor(False)
    state.model = SimpleNamespace(
        model=SimpleNamespace(
            get_codebook_offsets=lambda device: torch.zeros(8, dtype=torch.long, device=device),
            audio_embedding=lambda frame: frame[:, None].float().expand(8, 4),
        )
    )
    state.graph = lambda hidden: torch.arange(8)
    state.reset("sequential")
    return state


def step(state, token=0):
    logits = torch.zeros(1, 256)
    logits[0, token] = 1
    return state.step(torch.zeros(4), logits)


def test_sequential_audio_feedback_and_eos(state):
    step(state, 128)
    logits, event = step(state)
    assert logits.argmax().item() == 133
    assert event == [3, *range(8)]
    embeddings = state.embed(torch.tensor([133, 10]), torch.zeros(2, 4))
    assert embeddings[0].tolist() == [28] * 4
    assert embeddings[1].tolist() == [0] * 4
    state.graph = lambda hidden: torch.tensor([2048, 1, 2, 3, 4, 5, 6, 7])
    step(state)
    assert state.modality == 1 and state.events[-1] == [3, *([2048] * 8)]
    assert state.use_audio.item()  # The EOS audio frame is still the next backbone input.
    step(state, 7)
    result = output_tensors(state.events, device="cpu")
    assert result.sequences.tolist() == [[128, 7]]
    assert result.audio_codes.shape == (1, 8, 2)
    assert result.modalities.tolist() == [[1, 3, 3, 1]]
    assert not state.active and not state.use_audio.item()


def test_interleaving_and_text_end(state):
    state.reset("interleaved")
    step(state, 10)
    step(state, 11)
    assert state.modality == 3
    step(state)
    step(state)
    assert state.modality == 1
    step(state, 130)
    assert state.text_done and state.modality == 3
    for _ in range(5):
        step(state)
    assert state.modality == 3


def test_interleaved_eos_and_reset_keep_graph_buffers(state):
    pointer = state.next_embedding.data_ptr()
    step(state, 128)
    step(state)
    state.reset("interleaved")
    assert state.events == [] and not state.use_audio.item()
    assert state.next_embedding.data_ptr() == pointer
    _, event = step(state, 7)
    assert event is None and state.events == [] and not state.active
    result = output_tensors(state.events, device="cpu")
    assert result.audio_codes.shape == (1, 8, 0)


@pytest.fixture
def generator():
    generator = object.__new__(NativeGenerator)
    generator.closed = False
    generator.lock = threading.Lock()
    generator.max_cache_len = 16
    generator.frontend = Mock()
    generator.frontend.model._prepare_inputs_embeds.return_value = (torch.zeros(1, 3, 4), None, None)
    generator.engine = Mock()
    return generator


def test_native_generation_rejects_unsupported_options(generator):
    for options in (
        {"audio_top_k": 4, "audio_temperature": 0.8},
        {"generation_mode": "unknown"},
        {"max_new_tokens": 0},
        {"max_new_tokens": 1.5},
        {"attention_mask": torch.tensor([[1, 0]])},
        {"max_new_tokens": 14},
    ):
        with pytest.raises(ValueError):
            generator.generate(**options)
    generator.engine.generate.assert_not_called()
    generator.frontend.model._prepare_inputs_embeds.return_value = (torch.zeros(2, 3, 4), None, None)
    with pytest.raises(ValueError, match="batch size"):
        generator.generate(max_new_tokens=1)


def test_native_request_lock_is_released_after_errors(generator):
    generator.lock.acquire()
    with pytest.raises(RuntimeError, match="already serving"):
        generator.generate(max_new_tokens=1)
    generator.lock.release()
    generator.engine.generate.side_effect = RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        generator.generate(max_new_tokens=1)
    assert not generator.lock.locked()
    generator.close()
    generator.close()
    generator.engine.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        generator.generate(max_new_tokens=1)


def test_export_never_overwrites_source_or_existing_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for target in (source, source / "nested", tmp_path):
        with pytest.raises(ValueError, match="destination"):
            export_backbone(source, target)
    monkeypatch.setenv("FAST_LFM_BACKBONE", str(source))
    with pytest.raises(ValueError, match="Incomplete"):
        backbone_path(source)
    assert list(source.iterdir()) == []


def test_plugins_are_opt_in_and_backend_validation_precedes_loading(monkeypatch):
    monkeypatch.delenv("FAST_LFM_NATIVE_BACKEND", raising=False)
    register_vllm()
    register_sglang()
    with pytest.raises(ValueError, match="backend"):
        Pipeline(backend="unknown")


def test_pipeline_context_closes_native_workers():
    pipeline = object.__new__(Pipeline)
    pipeline.backend = "vllm"
    pipeline.model = Mock()
    with pytest.raises(RuntimeError, match="request failed"), pipeline:
        raise RuntimeError("request failed")
    pipeline.model.close.assert_called_once()


def test_decoder_setup_failure_closes_native_workers(monkeypatch):
    from fast_lfm_audio import pipeline as module
    from fast_lfm_audio.backends import model as native

    worker = Mock()
    monkeypatch.setattr(module, "model_path", lambda _: ".")
    monkeypatch.setattr(module.Lfm2AudioProcessor, "from_pretrained", Mock())
    monkeypatch.setattr(native, "NativeGenerator", Mock(return_value=worker))
    monkeypatch.setattr(module, "MimiDecoder", Mock(side_effect=RuntimeError("decoder failed")))
    with pytest.raises(RuntimeError, match="decoder failed"):
        Pipeline(backend="vllm")
    worker.close.assert_called_once()
