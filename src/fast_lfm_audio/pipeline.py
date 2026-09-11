"""Prompt preparation and waveform decoding for the optimized model."""

from pathlib import Path

import fast_mimi
import torch
from huggingface_hub import snapshot_download
from transformers import Lfm2AudioForConditionalGeneration, Lfm2AudioProcessor, MimiModel

from .runtime import optimize

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
MODEL_REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"
MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"
VOICES = ("US male", "US female", "UK male", "UK female")


def model_path(model_id=MODEL_ID):
    if Path(model_id).is_dir():
        return str(Path(model_id).resolve())
    return snapshot_download(model_id, revision=MODEL_REVISION if model_id == MODEL_ID else None)


def prepare_inputs(processor, *, prompt, text=None, audio=None):
    """Build the same GPU-preprocessed request for inference and benchmarks."""
    if (text is None) == (audio is None):
        raise ValueError("Provide exactly one of text or audio.")
    if text is not None and not text.strip():
        raise ValueError("Text must not be empty.")
    content = [{"type": "audio", "path": str(audio)}] if audio is not None else text
    return processor.apply_chat_template(
        [{"role": "system", "content": prompt}, {"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs={"audio_kwargs": {"device": "cuda"}},
    )


def decodable_codes(codes):
    if codes.ndim != 3 or codes.shape[:2] != (1, 8):
        raise ValueError("Expected audio codes with shape (1, 8, frames).")
    if codes.shape[-1] and bool((codes[..., -1] == 2048).all()):
        codes = codes[..., :-1]
    if codes.shape[-1] == 0:
        return codes
    if bool(((codes < 0) | (codes >= 2048)).any()):
        raise ValueError("Audio codes must be in [0, 2047], with an optional terminal EOS frame.")
    return codes.contiguous()


class MimiDecoder:
    """Decode eight codebooks, reusing graphs for power-of-two frame lengths."""

    def __init__(self, *, optimized=True, dtype="fp16", device="cuda", bucketed=True):
        self.model = MimiModel.from_pretrained("kyutai/mimi", revision=MIMI_REVISION).to(device).eval()
        self.bucketed = optimized and bucketed
        self.samples_per_frame = round(self.model.config.sampling_rate / self.model.config.frame_rate)
        if optimized:
            fast_mimi.optimize(self.model, dtype=dtype)

    @torch.inference_mode()
    def __call__(self, codes):
        frames = codes.shape[-1]
        if frames == 0:
            return torch.empty((1, 0), device=codes.device)
        if self.bucketed:
            # Mimi is causal: right-padding cannot affect preceding real samples.
            bucket = max(16, 1 << (frames - 1).bit_length())
            codes = torch.nn.functional.pad(codes, (0, bucket - frames))
        # Extra kwargs (including return_dict=True) trigger fast-mimi's eager fallback.
        return self.model.decode(codes).audio_values[:, 0, : frames * self.samples_per_frame].clone()

    @torch.inference_mode()
    def warmup(self, frame_lengths=(64, 128, 256, 512)):
        for frames in frame_lengths:
            self(torch.zeros((1, 8, frames), device=self.model.device, dtype=torch.long))
        torch.cuda.synchronize(self.model.device)


class Pipeline:
    """Load once, then call with text or audio for one request at a time."""

    def __init__(
        self,
        model_id=MODEL_ID,
        *,
        backend="transformers",
        codec="fast-mimi",
        backbone=True,
        max_cache_len=2048,
    ):
        if backend not in ("transformers", "vllm", "sglang"):
            raise ValueError("backend must be 'transformers', 'vllm', or 'sglang'.")
        if codec not in ("fast-mimi", "lfm"):
            raise ValueError("codec must be 'fast-mimi' or 'lfm'.")
        path = model_path(model_id)
        self.processor = Lfm2AudioProcessor.from_pretrained(path)
        self.backend = backend
        if backend == "transformers":
            self.model = Lfm2AudioForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, device_map="cuda"
            ).eval()
            self.optimization = optimize(self.model, backbone=backbone, max_cache_len=max_cache_len)
        else:
            from .backends.model import NativeGenerator

            self.model = NativeGenerator(path, backend, max_cache_len, graphs=backbone)
        try:
            self.decoder = MimiDecoder() if codec == "fast-mimi" else self.processor.decode_audio
        except BaseException:
            self.close()
            raise

    def __call__(self, *, text=None, audio=None, task="tts", voice="UK female", **kwargs):
        """Return (text, 24 kHz mono waveform, raw output); use greedy sampling by default."""
        inputs = self.prepare(text=text, audio=audio, task=task, voice=voice)
        options = {
            "generation_mode": "interleaved" if task == "chat" else "sequential",
            "max_new_tokens": 512,
            "text_top_k": 1,
            "audio_top_k": 1,
            "audio_temperature": 0.0,
        }
        options.update(kwargs)
        return self.generate(inputs, **options)

    def prepare(self, *, text=None, audio=None, task="tts", voice="UK female"):
        """Prepare TTS, spoken chat, or transcription inputs."""
        if task not in ("tts", "chat", "asr"):
            raise ValueError("task must be 'tts', 'chat', or 'asr'.")
        if task == "tts" and (audio is not None or voice not in VOICES):
            raise ValueError("TTS requires text and one of the four supported voices.")
        if task == "asr" and audio is None:
            raise ValueError("ASR requires audio.")
        prompt = {
            "tts": f"Perform TTS. Use the {voice} voice.",
            "chat": "Respond with interleaved text and audio.",
            "asr": "Perform ASR.",
        }[task]
        return prepare_inputs(self.processor, prompt=prompt, text=text, audio=audio)

    @torch.inference_mode()
    def generate(self, inputs, **kwargs):
        """Return (text, 24 kHz mono waveform, raw generation output)."""
        output = self.model.generate(**inputs.to("cuda"), **kwargs)
        codes = decodable_codes(output.audio_codes)
        waveform = self.decoder(codes) if codes.shape[-1] else torch.empty((1, 0), device="cuda")
        text = self.processor.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return text, waveform, output

    def close(self):
        if self.backend != "transformers":
            self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
