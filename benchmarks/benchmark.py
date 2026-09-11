"""Run each implementation in its own process, with identical requests and greedy sampling."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Lfm2AudioDetokenizer, Lfm2AudioForConditionalGeneration, Lfm2AudioProcessor

from fast_lfm_audio import optimize
from fast_lfm_audio.pipeline import MODEL_REVISION, MimiDecoder, decodable_codes, model_path, prepare_inputs

from .cases import CASES, ROOT


def metadata():
    return {
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {
            name: version(name) for name in ("transformers", "fast-mimi", "liquid-audio", "torchaudio")
        },
        "commits": {
            name: subprocess.check_output(
                ["git", "-C", str(ROOT / "vendor" / name), "rev-parse", "HEAD"], text=True
            ).strip()
            for name in ("transformers", "fast-mimi", "liquid-audio")
        },
        "model_revision": MODEL_REVISION,
        "cpu_threads": torch.get_num_threads(),
        "dtype": "bfloat16",
        "text_top_k": 1,
        "audio_top_k": 1,
    }


def digest(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


class Runner:
    def __init__(self, engine, max_cache_len):
        self.engine = engine
        path = model_path()
        self.first_audio_ms = None
        if engine == "liquid":
            from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

            self.processor = LFM2AudioProcessor.from_pretrained(Path(path)).eval()
            self.model = LFM2AudioModel.from_pretrained(Path(path)).eval()
            self.detokenizer = self.processor.audio_detokenizer
            self.decoder = self.processor.decode
        else:
            self.processor = Lfm2AudioProcessor.from_pretrained(path)
            self.model = Lfm2AudioForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, device_map="cuda"
            ).eval()
            if engine in ("depth", "fast"):
                self.optimization = optimize(
                    self.model, backbone=engine == "fast", max_cache_len=max_cache_len
                )
            if engine == "fast":
                self.decoder = MimiDecoder()
            else:
                self.processor._detokenizer = (
                    Lfm2AudioDetokenizer.from_pretrained(
                        path, subfolder="audio_detokenizer", dtype=torch.float32
                    )
                    .cuda()
                    .eval()
                )
                self.decoder = self.processor.decode_audio

        original = self.model._sample_audio_frame

        def timed_sample(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.first_audio_ms is None:
                torch.cuda.synchronize()
                self.first_audio_ms = (time.perf_counter() - self.generation_start) * 1000
            return result

        self.model._sample_audio_frame = timed_sample

    def prepare(self, case):
        if self.engine == "liquid":
            from liquid_audio import ChatState

            chat = ChatState(self.processor)
            chat.new_turn("system")
            chat.add_text(case["prompt"])
            chat.end_turn()
            chat.new_turn("user")
            if "audio" in case:
                wave, rate = sf.read(ROOT / case["audio"], dtype="float32")
                chat.add_audio(torch.from_numpy(wave).unsqueeze(0), rate)
            else:
                chat.add_text(case["text"])
            chat.end_turn()
            chat.new_turn("assistant")
            return dict(chat)
        return prepare_inputs(
            self.processor,
            prompt=case["prompt"],
            text=case.get("text"),
            audio=ROOT / case["audio"] if "audio" in case else None,
        ).to("cuda")

    def generate(self, inputs, case, max_tokens):
        self.first_audio_ms = None
        self.generation_start = time.perf_counter()
        options = {"max_new_tokens": max_tokens, "text_top_k": 1, "audio_top_k": 1}
        mode = case.get("mode", "sequential")
        if self.engine != "liquid":
            result = self.model.generate(**inputs, generation_mode=mode, **options)
            return {name: getattr(result, name) for name in ("sequences", "audio_codes", "modalities")}
        events = list(getattr(self.model, f"generate_{mode}")(**inputs, **options))
        text = [event for event in events if event.numel() == 1]
        audio = [event for event in events if event.numel() == 8]
        return {
            "sequences": torch.cat(text)[None]
            if text
            else torch.empty((1, 0), device="cuda", dtype=torch.long),
            "audio_codes": torch.stack(audio, dim=-1)[None]
            if audio
            else torch.empty((1, 8, 0), device="cuda", dtype=torch.long),
            "modalities": torch.tensor([[1 if event.numel() == 1 else 3 for event in events]], device="cuda"),
        }

    def run(self, case, max_tokens):
        torch.cuda.synchronize()
        start = time.perf_counter()
        inputs = self.prepare(case)
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        output = self.generate(inputs, case, max_tokens)
        torch.cuda.synchronize()
        generated = time.perf_counter()
        codes = decodable_codes(output["audio_codes"])
        wave = self.decoder(codes) if codes.shape[-1] else torch.empty((1, 0), device="cuda")
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings = {
            "prepare_ms": (prepared - start) * 1000,
            "generation_ms": (generated - prepared) * 1000,
            "decode_ms": (end - generated) * 1000,
            "total_ms": (end - start) * 1000,
            "first_audio_token_ms": self.first_audio_ms,
            "audio_seconds": wave.shape[-1] / 24000,
        }
        return timings, output, wave


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("liquid", "hf", "depth", "fast"), required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-cache-len", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    cases = json.loads(args.cases_file.read_text()) if args.cases_file else CASES
    selected = args.cases or list(cases)
    if args.repeats < 1 or not selected or any(name not in cases for name in selected):
        parser.error("Select valid cases and at least one measured repeat.")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    runner = Runner(args.engine, args.max_cache_len)
    torch.cuda.synchronize()
    report = {
        "engine": args.engine,
        "setup_seconds": time.perf_counter() - start,
        "environment": metadata(),
        "max_new_tokens": args.max_new_tokens,
        "max_cache_len": args.max_cache_len,
        "cases": {},
    }
    artifacts = {}
    for name in selected:
        case = cases[name]
        max_tokens = case.get("max_new_tokens", args.max_new_tokens)
        torch.cuda.reset_peak_memory_stats()
        cold, expected, _ = runner.run(case, max_tokens)
        print(
            f"{args.engine} {name} first request: total={cold['total_ms']:.1f} ms "
            f"audio={cold['audio_seconds']:.2f} s",
            flush=True,
        )
        expected = {key: tensor.cpu() for key, tensor in expected.items()}
        samples = []
        repeated_exact = True
        for index in range(args.repeats):
            sample, output, wave = runner.run(case, max_tokens)
            if not bool(torch.isfinite(wave).all()):
                raise RuntimeError(f"Non-finite waveform: {name}")
            if (
                "target_frames" in case
                and decodable_codes(output["audio_codes"]).shape[-1] != case["target_frames"]
            ):
                raise RuntimeError(f"TTS stopped before the requested frame budget: {name}")
            samples.append(sample)
            cpu_output = {key: tensor.cpu() for key, tensor in output.items()}
            repeated_exact &= all(torch.equal(expected[key], tensor) for key, tensor in cpu_output.items())
            print(
                f"{args.engine} {name} {index + 1}/{args.repeats}: "
                f"generation={sample['generation_ms']:.1f} ms total={sample['total_ms']:.1f} ms "
                f"audio={sample['audio_seconds']:.2f} s",
                flush=True,
            )
        medians = {
            key: statistics.median([sample[key] for sample in samples])
            for key in samples[0]
            if samples[0][key] is not None
        }
        medians["rtf"] = (
            medians["total_ms"] / (1000 * medians["audio_seconds"]) if medians["audio_seconds"] else None
        )
        medians["events_per_second"] = output["modalities"].numel() * 1000 / medians["generation_ms"]
        report["cases"][name] = {
            "request": case,
            "max_new_tokens": max_tokens,
            "cold": cold,
            "samples": samples,
            "median": medians,
            "p95_total_ms": float(np.percentile([sample["total_ms"] for sample in samples], 95)),
            "events": output["modalities"].numel(),
            "audio_frames": output["audio_codes"].shape[-1],
            "limit_reached": output["modalities"].numel() == max_tokens,
            "repeat_exact": repeated_exact,
            "hashes": {key: digest(tensor) for key, tensor in cpu_output.items()},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        artifacts[name] = cpu_output
        torch.save(artifacts, args.output_dir / f"{args.engine}_tokens.pt")
        sf.write(args.output_dir / f"{args.engine}_{name}.wav", wave[0].float().cpu().numpy(), 24000)
        (args.output_dir / f"{args.engine}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
