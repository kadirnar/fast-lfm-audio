"""Prompt preparation and waveform decoding for the optimized model."""

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import fast_mimi
import torch
from huggingface_hub import snapshot_download
from transformers import Lfm2AudioForConditionalGeneration, Lfm2AudioProcessor, MimiModel

from .runtime import GraphedCall, optimize
from .strict import ExactModules, fp32_precision

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
MODEL_REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"
MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"
VOICES = ("US male", "US female", "UK male", "UK female")


def model_path(model_id=MODEL_ID):
    if Path(model_id).is_dir():
        return str(Path(model_id).resolve())
    return snapshot_download(model_id, revision=MODEL_REVISION if model_id == MODEL_ID else None)


def prepare_inputs(processor, *, prompt, text=None, audio=None):
    """Build a GPU-preprocessed request for text or audio inference."""
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
    """Decode eight codebooks with exact frame lengths in strict FP32 mode."""

    def __init__(self, *, optimized=True, dtype="fp16", device="cuda", bucketed=True, strict=None):
        if strict is None:
            strict = dtype in ("fp32", "float32", torch.float32)
        if strict and dtype not in ("fp32", "float32", torch.float32):
            raise ValueError("Strict Mimi decoding requires dtype='fp32'.")
        self.model = (
            MimiModel.from_pretrained("kyutai/mimi", revision=MIMI_REVISION, dtype=torch.float32)
            .to(device)
            .eval()
        )
        self.strict = strict
        self.optimized = optimized
        self.graphs = OrderedDict()
        self.prefix_graphs = OrderedDict()
        self.exact_modules = ExactModules(self.model) if optimized and strict else None
        self.bucketed = optimized and bucketed and not strict
        self.samples_per_frame = round(self.model.config.sampling_rate / self.model.config.frame_rate)
        if optimized and not strict:
            fast_mimi.optimize(self.model, dtype=dtype)

    @torch.inference_mode()
    def __call__(self, codes):
        frames = codes.shape[-1]
        if frames == 0:
            return torch.empty((1, 0), device=codes.device)
        if self.strict:
            with fp32_precision():
                if not self.optimized or not codes.is_cuda or self.model.training:
                    waveform = self.model.decode(codes).audio_values
                else:
                    key = (codes.shape, codes.stride(), codes.dtype, codes.device)
                    if key not in self.graphs:
                        # Exact frame lengths retain convolution and attention shapes.
                        # Bound retained graph memory for requests with varying lengths.
                        if len(self.graphs) >= 2:
                            self.graphs.popitem(last=False)
                        self.graphs[key] = GraphedCall(
                            lambda value: self.model.decode(value).audio_values, codes
                        )
                    self.graphs.move_to_end(key)
                    waveform = self.graphs[key](codes)
                return waveform[:, 0, : frames * self.samples_per_frame].clone()
        if self.bucketed:
            # Right-padding retains causal context. Different bucket shapes can
            # still change floating-point rounding, so strict mode never pads.
            bucket = max(16, 1 << (frames - 1).bit_length())
            codes = torch.nn.functional.pad(codes, (0, bucket - frames))
        # Extra kwargs (including return_dict=True) trigger fast-mimi's eager fallback.
        return self.model.decode(codes).audio_values[:, 0, : frames * self.samples_per_frame].clone()

    @torch.inference_mode()
    def decode_prefix(self, codes):
        """Decode available codes for early delivery, retaining FP32 arithmetic.

        Padding changes reference operation shapes and can change waveform bits.
        This entry point is used only when prefix streaming is requested; normal
        strict decoding continues to use the exact, unpadded frame count.
        """
        if not self.strict or not codes.shape[-1]:
            return self(codes)
        frames = codes.shape[-1]
        bucket = max(16, 1 << (frames - 1).bit_length())
        padded = torch.nn.functional.pad(codes, (0, bucket - frames))
        with fp32_precision():
            if not self.optimized or not padded.is_cuda or self.model.training:
                waveform = self.model.decode(padded).audio_values
            else:
                key = (padded.shape, padded.stride(), padded.dtype, padded.device)
                if key not in self.prefix_graphs:
                    # Keep the first-audio bucket resident across long requests.
                    # Two additional shapes bound retained decoder graph memory.
                    if len(self.prefix_graphs) >= 3:
                        victim = next(k for k in self.prefix_graphs if k[0][-1] != 16)
                        self.prefix_graphs.pop(victim).close()
                    self.prefix_graphs[key] = GraphedCall(
                        lambda value: self.model.decode(value).audio_values, padded
                    )
                self.prefix_graphs.move_to_end(key)
                waveform = self.prefix_graphs[key](padded)
            return waveform[:, 0, : frames * self.samples_per_frame].clone()

    @torch.inference_mode()
    def warmup(self, frame_lengths=(64, 128, 256, 512)):
        for frames in frame_lengths:
            self(torch.zeros((1, 8, frames), device=self.model.device, dtype=torch.long))
        torch.cuda.synchronize(self.model.device)

    def close(self):
        self.graphs.clear()
        self.prefix_graphs.clear()
        if self.exact_modules is not None:
            self.exact_modules.close()
        if self.strict:
            self.optimized = False


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
        dtype="fp32",
        strict=None,
    ):
        if backend not in ("transformers", "vllm", "sglang"):
            raise ValueError("backend must be 'transformers', 'vllm', or 'sglang'.")
        if codec not in ("fast-mimi", "lfm"):
            raise ValueError("codec must be 'fast-mimi' or 'lfm'.")
        dtypes = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtypes.get(dtype, dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be 'fp32', 'fp16', or 'bf16'.")
        self.strict = dtype == torch.float32 if strict is None else strict
        if self.strict and dtype != torch.float32:
            raise ValueError("Strict mode requires dtype='fp32'.")
        if backend != "transformers" and dtype != torch.bfloat16:
            raise ValueError(
                "Strict FP32 uses backend='transformers'. Native audio adapters currently require dtype='bf16'."
            )
        if dtype == torch.float32 and not self.strict:
            raise ValueError("The FP32 pipeline requires strict=True.")
        path = model_path(model_id)
        self.processor = Lfm2AudioProcessor.from_pretrained(path)
        self.backend = backend
        if backend == "transformers":
            self.model = Lfm2AudioForConditionalGeneration.from_pretrained(
                path, dtype=dtype, device_map="cuda"
            ).eval()
            self.optimization = optimize(
                self.model, backbone=backbone, max_cache_len=max_cache_len, strict=self.strict
            )
        else:
            from .backends.model import NativeGenerator

            self.model = NativeGenerator(path, backend, max_cache_len, graphs=backbone)
        try:
            self.decoder = (
                MimiDecoder(dtype="fp32" if self.strict else "fp16", strict=self.strict)
                if codec == "fast-mimi"
                else self.processor.decode_audio
            )
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
    def generate(self, inputs, *, on_audio=None, chunk_frames=4, streaming_mode="prefix", **kwargs):
        """Return (text, 24 kHz mono waveform, raw generation output).

        Supply ``on_audio(cpu_float32_chunk)`` to receive owned chunks of shape
        (1, samples). Prefixes are delivered during generation, including FP32.
        FP32 arithmetic and model tokens are retained, but early decoder shapes
        can change waveform rounding. Use ``streaming_mode="buffered"`` to wait
        for the exact full waveform before sending chunks.
        Chunks concatenate to the returned waveform. The callback should enqueue
        promptly: blocking playback also blocks this request.
        """
        stream = None
        if on_audio is not None:
            if self.backend != "transformers" or not isinstance(self.decoder, MimiDecoder):
                raise ValueError("Audio streaming requires backend='transformers' and codec='fast-mimi'.")
            if not callable(on_audio):
                raise TypeError("on_audio must be callable.")
            if type(chunk_frames) is not int or chunk_frames < 1:
                raise ValueError("chunk_frames must be a positive integer.")
            if streaming_mode not in ("prefix", "buffered"):
                raise ValueError("streaming_mode must be 'prefix' or 'buffered'.")
            from .streaming import BufferedAudioStream, PrefixAudioStream

            stream = (
                BufferedAudioStream(on_audio, chunk_frames * self.decoder.samples_per_frame)
                if streaming_mode == "buffered"
                else PrefixAudioStream(
                    self.decoder.decode_prefix if self.strict else self.decoder,
                    on_audio,
                    chunk_frames,
                    self.model.config.audio_eos_token_id,
                )
            )
        sample_prefixes = stream is not None and streaming_mode == "prefix"
        with fp32_precision() if self.strict else nullcontext():
            original_sample = self.model._sample_audio_frame if sample_prefixes else None
            if sample_prefixes:

                def sample(*args, **options):
                    frame = original_sample(*args, **options)
                    stream.append(frame)
                    return frame

                self.model._sample_audio_frame = sample
            try:
                output = self.model.generate(**inputs.to("cuda"), **kwargs)
            finally:
                if sample_prefixes:
                    self.model._sample_audio_frame = original_sample
            codes = decodable_codes(output.audio_codes)
            if sample_prefixes:
                waveform = stream.finish_codes(codes, self.decoder.samples_per_frame)
            else:
                waveform = self.decoder(codes) if codes.shape[-1] else torch.empty((1, 0), device="cuda")
                if stream is not None:
                    waveform = stream.finish(waveform)
            text = self.processor.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            return text, waveform, output

    def close(self):
        if self.backend != "transformers":
            self.model.close()
        elif hasattr(self, "optimization") and hasattr(self.model, "_fast_lfm_optimization"):
            self.optimization.close()
        close_decoder = getattr(getattr(self, "decoder", None), "close", None)
        if close_decoder is not None:
            close_decoder()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
