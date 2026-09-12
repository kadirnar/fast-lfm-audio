import argparse
from pathlib import Path

import soundfile as sf
import torch

from .pipeline import VOICES, Pipeline


def main():
    parser = argparse.ArgumentParser(description="LFM2.5 Audio inference with CUDA graphs and fast-mimi")
    parser.add_argument("--task", choices=("tts", "chat", "asr"), default="tts")
    parser.add_argument("--backend", choices=("transformers", "vllm", "sglang"), default="transformers")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text")
    source.add_argument("--audio", type=Path)
    parser.add_argument("--voice", choices=VOICES, default="UK female")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--codec", choices=("fast-mimi", "lfm"), default="fast-mimi")
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="FP32 uses strict Triton fusion; FP16/BF16 use reduced-precision inference.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-cache-len", type=int, default=2048)
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--audio-temperature", type=float, default=0.0)
    parser.add_argument("--audio-top-k", type=int, default=1)
    args = parser.parse_args()
    if args.backend != "transformers" and args.dtype != "bf16":
        parser.error("Native audio adapters require --dtype bf16; strict FP32 uses --backend transformers")
    if args.task == "asr" and args.audio is None:
        parser.error("--audio is required for ASR")
    if args.task == "tts" and args.audio is not None:
        parser.error("TTS takes --text; --audio is supported for chat and ASR")
    if args.audio is not None and not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")
    if args.text is not None and not args.text.strip():
        parser.error("--text must not be empty")
    if args.audio is None and args.text is None:
        args.text = "Hello, this is a test of fast audio generation."
    torch.set_num_threads(4 if args.backend == "transformers" else 1)
    with Pipeline(
        backend=args.backend,
        codec=args.codec,
        backbone=not args.depth_only,
        max_cache_len=args.max_cache_len,
        dtype=args.dtype,
    ) as pipeline:
        text, waveform, output = pipeline(
            task=args.task,
            voice=args.voice,
            text=args.text,
            audio=args.audio,
            max_new_tokens=args.max_new_tokens,
            audio_top_k=args.audio_top_k,
            audio_temperature=args.audio_temperature,
        )
    if text:
        print(text)
    if waveform.numel():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            args.output,
            waveform[0].float().cpu().numpy(),
            24000,
            subtype="FLOAT" if args.dtype == "fp32" else None,
        )
        print(f"Audio: {args.output.resolve()} ({waveform.shape[-1] / 24000:.2f} s)")
    if output.modalities.shape[-1] == args.max_new_tokens:
        print("Token limit reached; output may be incomplete.")


if __name__ == "__main__":
    main()
