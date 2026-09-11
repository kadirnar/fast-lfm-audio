"""Run the three backends sequentially and report complete-waveform latency."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from .cases import ROOT, make_duration_cases
from .report import compare_outputs, table, waveform_statistics

BACKENDS = {"fast": "Transformers", "vllm": "vLLM", "sglang": "SGLang"}


def write_report(directory):
    data = {engine: json.loads((directory / f"{engine}.json").read_text()) for engine in BACKENDS}
    reference = data["fast"]
    tokens, parity = {}, {}
    waveform_path = directory / "waveforms.json"
    waveforms = json.loads(waveform_path.read_text()) if waveform_path.exists() else {}
    for engine, report in data.items():
        if not report["cases"] or set(report["cases"]) != set(reference["cases"]):
            raise ValueError(f"Different case sets: {engine}")
        if report["max_cache_len"] != reference["max_cache_len"]:
            raise ValueError(f"Different cache limits: {engine}")
        for key in (
            "gpu",
            "torch",
            "cuda",
            "model_revision",
            "commits",
            "cpu_threads",
            "dtype",
            "text_top_k",
            "audio_top_k",
        ):
            if report["environment"][key] != reference["environment"][key]:
                raise ValueError(f"Different benchmark environments: {engine}: {key}")
        path = directory / f"{engine}_tokens.pt"
        tokens[engine] = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else {}
        for name, case in report["cases"].items():
            expected = reference["cases"][name]
            if case["request"] != expected["request"] or case["max_new_tokens"] != expected["max_new_tokens"]:
                raise ValueError(f"Different requests: {engine}: {name}")
            if not case["samples"] or len(case["samples"]) != len(expected["samples"]):
                raise ValueError(f"Different repeat counts: {engine}: {name}")
            if engine != "fast":
                parity.setdefault(name, {})[engine] = compare_outputs(
                    expected, case, tokens["fast"].get(name), tokens[engine].get(name)
                )
            path = directory / f"{engine}_{name}.wav"
            if path.exists():
                waveforms.setdefault(name, {})[engine] = waveform_statistics(path)

    environment = reference["environment"]
    repeats = len(next(iter(reference["cases"].values()))["samples"])
    lines = [
        "# Three-Backend Benchmark",
        "",
        f"{environment['gpu']}; PyTorch {environment['torch']}; CUDA {environment['cuda']}; BF16; batch one.",
        f"Each case: one first request, then {repeats} warm requests. Tables show warm medians in seconds.",
        "End-to-end = input preparation + model generation + complete fast-mimi decoding, with CUDA synchronization.",
        "Includes native-engine IPC. Excludes model loading, initial setup, file writing, recording, and playback.",
        "All engines run separately. This is an offline, single-request comparison, not a serving-throughput test.",
        "",
        "Transformers uses this repo's backbone/depthformer CUDA graphs. The other columns use this repo's",
        "experimental adapters: native LFM2 backbone kernels and caches, the PR's audio encoder/depthformer, and fast-mimi.",
        "They are not upstream native LFM2-Audio implementations, and never fall back to a Transformers backbone.",
        "",
    ]
    lines += table(
        "Versions",
        ["Backend", "Library", "Python", "Model Setup (s)"],
        [
            [
                label,
                report["environment"]["packages"]["transformers" if engine == "fast" else engine],
                report["environment"]["python"],
                f"{report['setup_seconds']:.2f}",
            ]
            for engine, label in BACKENDS.items()
            for report in (data[engine],)
        ],
    )
    lines += [
        "SGLang 0.5.19 pins Transformers 5.12.1 and tokenizers 0.22.2. This adapter explicitly overrides",
        "them with the pinned audio PR (5.16.0.dev0) and tokenizers 0.23.2, with a narrow config-registration shim.",
        "Its environment therefore has two declared dependency conflicts despite passing these inference tests.",
        "Native decoding uses CUDA graphs; vLLM uses FlashAttention 2 and SGLang uses FA4 on this RTX 5070 Ti.",
        "The isolated JIT toolkit is CUDA 13.4; Torch's runtime stays CUDA 13.0. No upstream source files are edited.",
        "",
    ]
    for task, title, first_column in (
        ("tts", "Text to Speech", "Audio Budget"),
        ("chat", "Voice Chat", "Input Audio"),
    ):
        rows = []
        for name, case in reference["cases"].items():
            if case["request"].get("task") != task:
                continue
            seconds = case["request"]["target_seconds" if task == "tts" else "input_seconds"]
            cells = []
            for engine in BACKENDS:
                result = data[engine]["cases"][name]
                median = result["median"]
                partial = " (capped)" if result["limit_reached"] else ""
                cells.append(
                    f"**{median['total_ms'] / 1000:.3f} s**<br>"
                    f"<sub>{median['audio_seconds']:.2f} s audio{partial}</sub>"
                )
            rows.append([f"{seconds} s", *cells])
        lines += table(title, [first_column, *BACKENDS.values()], rows)
    lines += [
        "**TTS uses fixed event budgets, not sentence completion.** 63/250/1250 frames produce 5.04/20/100 s.",
        "**The 100 s case is a stress test, not 100 s of continuous speech.** Long low-signal tails occur in all three outputs.",
        "Chat inputs tile/crop the upstream 4.904 s recording; these are synthetic length tests, not natural long conversations.",
        "Chat replies differ in content/length across backends. Raw chat latency is not an equal-output speedup comparison.",
        "Capped replies may be incomplete. Token equality is not a perceptual quality test.",
        "",
    ]
    rows = []
    for name in reference["cases"]:
        rows.append(
            [name, *["yes" if data[engine]["cases"][name]["repeat_exact"] else "no" for engine in BACKENDS]]
        )
    lines += table("Exact Tokens Across Repeats", ["Case", *BACKENDS.values()], rows)
    lines += table(
        "Exact Tokens vs Transformers",
        ["Case", "vLLM", "SGLang"],
        [
            [name, *["yes" if parity[name][engine]["all_exact"] else "no" for engine in ("vllm", "sglang")]]
            for name in reference["cases"]
        ],
    )
    lines += table(
        "100 s Waveform Audit",
        ["Backend", "Trailing Quiet Audio", "Finite Samples"],
        [
            [
                label,
                f"{waveforms.get('tts_100s', {}).get(engine, {}).get('trailing_quiet_seconds', 'unavailable')} s",
                str(waveforms.get("tts_100s", {}).get(engine, {}).get("finite", "unavailable")),
            ]
            for engine, label in BACKENDS.items()
        ],
    )
    lines += [
        "Quiet = one-second RMS below 0.001. Audio lengths and quiet tails do not establish intelligibility.",
        "Raw measurements: [Transformers](fast.json), [vLLM](vllm.json), [SGLang](sglang.json).",
        "First-request timings are in each JSON; disk JIT caches may already be warm.",
        "JSON memory counters cover the driver process only, not native engine workers; they are not total-engine VRAM.",
        "Validation: [token parity](parity.json), [waveform statistics](waveforms.json).",
        "",
        "Native implementations: [vLLM LFM2](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/lfm2.py),",
        "[SGLang LFM2](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/models/lfm2.py).",
        "",
    ]
    (directory / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    waveform_path.write_text(json.dumps(waveforms, indent=2) + "\n")
    (directory / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backends/inference")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
    if not args.report_only:
        for path in (".venv", "vendor/envs/vllm", "vendor/envs/sglang-py312"):
            if not (ROOT / path / "bin/python").exists():
                parser.error("Run scripts/setup.sh and scripts/backend.sh {vllm|sglang} --setup first.")
        make_duration_cases(ROOT / "results/durations")
        options = [
            "--cases-file",
            str(ROOT / "results/durations/cases.json"),
            "--repeats",
            str(args.repeats),
            "--max-cache-len",
            "4096",
            "--output-dir",
            str(args.output_dir.resolve()),
        ]
        commands = [
            [str(ROOT / ".venv/bin/python"), "-m", "benchmarks.benchmark", "--engine", "fast"],
            ["bash", "scripts/backend.sh", "vllm", "--benchmark"],
            ["bash", "scripts/backend.sh", "sglang", "--benchmark"],
        ]
        for command in commands:
            subprocess.run([*command, *options], cwd=ROOT, check=True)
    write_report(args.output_dir)
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
