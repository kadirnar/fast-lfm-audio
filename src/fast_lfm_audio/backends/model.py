"""Checkpoint export and the shared audio frontend for the native engines."""

import atexit
import hashlib
import json
import os
import sys
import tempfile
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from filelock import FileLock
from safetensors.torch import save_file
from transformers import Lfm2AudioForConditionalGeneration

from ..runtime import GraphedAudioEncoder
from .audio import output_tensors

VERSIONS = {"vllm": "0.29.0", "sglang": "0.5.19"}


def export_backbone(source, destination):
    """Write native HF LFM2 weights into a new directory; leave the source untouched."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Backbone export requires a new destination outside the source directory.")
    model = Lfm2AudioForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16).eval()
    destination.mkdir(parents=True)
    weights = {"model." + name: value.contiguous() for name, value in model.model.lfm.state_dict().items()}
    save_file(weights, destination / "model.safetensors")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (source / name).exists():
            (destination / name).symlink_to((source / name).resolve())
    config = model.config.text_config
    config.architectures = ["Lfm2ForCausalLM"]
    config.eos_token_id = model.config.eos_token_id
    config.fast_lfm_audio_source = str(source)
    config.save_pretrained(destination)


def backbone_path(source):
    source = str(Path(source).resolve())
    key = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = Path(os.environ.get("FAST_LFM_BACKBONE", Path.home() / ".cache/fast-lfm-audio" / key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
                staging = Path(temporary) / "backbone"
                export_backbone(source, staging)
                staging.rename(path)
        if not (path / "config.json").is_file():
            raise ValueError(f"Incomplete backbone export: {path}")
        config = json.loads((path / "config.json").read_text())
        if config.get("fast_lfm_audio_source") != source or not (path / "model.safetensors").is_file():
            raise ValueError(f"Incomplete or mismatched backbone export: {path}")
    return path.resolve()


class NativeGenerator:
    """Batch one, one CUDA device, greedy decoding. Never silently fall back to HF."""

    def __init__(self, source, backend, max_cache_len=4096, graphs=True):
        if backend not in VERSIONS:
            raise ValueError("backend must be 'vllm' or 'sglang'.")
        try:
            installed = version(backend)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Run: bash scripts/backend.sh {backend} --setup") from error
        if installed != VERSIONS[backend]:
            raise ValueError(f"This adapter requires {backend}=={VERSIONS[backend]}.")
        if not isinstance(max_cache_len, int) or not 16 <= max_cache_len <= 4096:
            raise ValueError("Native adapters require max_cache_len between 16 and 4096.")
        os.environ["FAST_LFM_NATIVE_BACKEND"] = backend
        toolkit = (
            Path(__file__).resolve().parents[3] / "vendor/envs/cuda/lib/python3.12/site-packages/nvidia/cu13"
        )
        if (toolkit / "bin/nvcc").is_file():
            os.environ.setdefault("CUDA_HOME", str(toolkit))
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        path = backbone_path(source)
        self.frontend = Lfm2AudioForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16).eval()
        self.frontend.model.lfm.layers = torch.nn.ModuleList()
        self.frontend = self.frontend.cuda()
        self.encoder = GraphedAudioEncoder(self.frontend.model) if graphs else None
        self.config = self.frontend.config
        self.max_cache_len = max_cache_len
        self.lock = threading.Lock()
        self.closed = False
        if backend == "vllm":
            from .vllm import VllmEngine

            self.engine = VllmEngine(path, max_cache_len, graphs)
        else:
            from .sglang import SGLangEngine

            self.engine = SGLangEngine(path, max_cache_len, graphs)
        atexit.register(self.close)

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        input_features_attention_mask=None,
        modality_ids=None,
        audio_codes=None,
        inputs_embeds=None,
        max_new_tokens=512,
        generation_mode="sequential",
        text_temperature=None,
        text_top_k=1,
        audio_temperature=None,
        audio_top_k=1,
        on_audio_frame=None,
    ):
        if self.closed:
            raise RuntimeError("The native engine is closed.")
        if on_audio_frame is not None and not callable(on_audio_frame):
            raise TypeError("on_audio_frame must be callable.")
        if (
            generation_mode not in ("sequential", "interleaved")
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("Use a valid generation mode and a positive token budget.")
        if any(
            k != 1 and t is not None and t > 0
            for k, t in ((text_top_k, text_temperature), (audio_top_k, audio_temperature))
        ):
            raise ValueError("Native audio adapters currently support greedy sampling only.")
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise ValueError("Padded prompts are not supported by the batch-one native adapters.")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("The native audio engine is already serving a request.")
        try:
            if self.closed:
                raise RuntimeError("The native engine is closed.")
            embeddings, _, _ = self.frontend.model._prepare_inputs_embeds(
                input_ids,
                attention_mask,
                input_features,
                input_features_attention_mask,
                modality_ids,
                audio_codes,
                inputs_embeds,
            )
            if embeddings.shape[0] != 1:
                raise ValueError("Native audio adapters require batch size one.")
            if embeddings.shape[1] + max_new_tokens > self.max_cache_len:
                raise ValueError("Prompt plus token budget exceeds max_cache_len.")
            options = {}
            if on_audio_frame is not None:

                def on_event(event):
                    if event is not None and event[0] == 3:
                        on_audio_frame(torch.tensor(event[1:], dtype=torch.long, device=embeddings.device))

                options["on_event"] = on_event
            return output_tensors(
                self.engine.generate(embeddings[0], generation_mode, max_new_tokens, **options)
            )
        finally:
            self.lock.release()

    def close(self):
        with self.lock:
            if not self.closed:
                self.engine.close()
                if getattr(self, "encoder", None) is not None:
                    self.encoder.close()
                self.closed = True
                atexit.unregister(self.close)
