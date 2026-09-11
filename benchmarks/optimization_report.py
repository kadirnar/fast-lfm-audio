"""Compare the same backend before and after a runtime optimization."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .compare_backends import BACKENDS
from .report import compare_outputs, table


def compare_waveforms(before, after):
    reference, rate = sf.read(before)
    candidate, candidate_rate = sf.read(after)
    if rate != candidate_rate or reference.shape != candidate.shape:
        raise ValueError("Different waveform shapes or sample rates.")
    difference = reference - candidate
    signal, noise = np.mean(reference**2), np.mean(difference**2)
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "max_abs_difference": float(np.abs(difference).max(initial=0)),
        "snr_db": float(10 * np.log10(signal / noise)) if signal > 0 and noise > 0 else None,
    }


def write_report(directory):
    rows, parity = {}, {}
    waveform_path = directory / "waveform-parity.json"
    waveforms = json.loads(waveform_path.read_text()) if waveform_path.exists() else {}
    for engine, label in BACKENDS.items():
        before, after = [
            json.loads((directory / stage / f"{engine}.json").read_text()) for stage in ("before", "after")
        ]
        if engine == "fast":
            environment = before["environment"]
        if before["environment"] != after["environment"] or before["max_cache_len"] != after["max_cache_len"]:
            raise ValueError(f"Different environments: {engine}")
        if set(before["cases"]) != set(after["cases"]):
            raise ValueError(f"Different case sets: {engine}")
        tokens = []
        for stage in ("before", "after"):
            path = directory / stage / f"{engine}_tokens.pt"
            tokens.append(torch.load(path, map_location="cpu", weights_only=True) if path.exists() else {})
        rows[engine], parity[engine] = [], {}
        for name, old in before["cases"].items():
            new = after["cases"][name]
            if old["request"] != new["request"] or old["max_new_tokens"] != new["max_new_tokens"]:
                raise ValueError(f"Different requests: {engine}: {name}")
            if not old["samples"] or len(old["samples"]) != len(new["samples"]):
                raise ValueError(f"Different repeats: {engine}: {name}")
            for case in (old, new):
                if case["median"]["total_ms"] != statistics.median(
                    sample["total_ms"] for sample in case["samples"]
                ):
                    raise ValueError(f"Invalid recorded median: {engine}: {name}")
            check = compare_outputs(old, new, tokens[0].get(name), tokens[1].get(name))
            parity[engine][name] = check
            paths = [directory / stage / f"{engine}_{name}.wav" for stage in ("before", "after")]
            if all(path.exists() for path in paths):
                waveforms.setdefault(engine, {})[name] = compare_waveforms(*paths)
            previous, current = old["median"]["total_ms"], new["median"]["total_ms"]
            rows[engine].append(
                [
                    name,
                    f"{previous / 1000:.3f} s",
                    f"{current / 1000:.3f} s",
                    f"{previous / current:.2f}x" if check["all_exact"] else "different output",
                    "yes" if check["all_exact"] else "no",
                ]
            )
    lines = [
        "# Runtime Optimization",
        "",
        "Same GPU, model weights, requests, greedy sampling, context limits, and repeat counts.",
        "End-to-end medians include input preparation, generation, full fast-mimi decoding, and CUDA synchronization.",
        "Model loading and first-use setup are excluded. Speedups are shown only for exact token matches.",
        "The shared greedy Depthformer now uses selective torch.compile fusion before CUDA graph capture.",
        "ATen mean reductions, eager rotary-buffer initialization, and BF16 rounding points are preserved.",
        "Weights, codebook counts, generation budgets, and sampling rules are unchanged; no quantization is used.",
        "First use includes compilation; disk compiler caches may already be warm in these measurements.",
        "",
    ]
    checks = [check for cases in waveforms.values() for check in cases.values()]
    if checks:
        lines += [
            f"Saved PCM files: {sum(check['exact'] for check in checks)}/{len(checks)} bitwise equal.",
            "Token equality does not imply bitwise PCM equality. [Waveform comparisons](waveform-parity.json).",
            "",
        ]
    for engine, label in BACKENDS.items():
        lines += table(label, ["Case", "Before", "After", "Speedup", "Exact Tokens"], rows[engine])
    lines += [
        "The 100 s TTS case contains long quiet tails and is a fixed-budget stress test, not continuous speech.",
        "Chat input audio is repeated/cropped from the upstream fixture. Some responses hit the token limit.",
        "Before/after token equality checks both content and length; it does not establish perceptual quality.",
        "",
        "[Before](before/REPORT.md) | [After](after/REPORT.md) | [Token checks](parity.json)",
        "",
    ]
    profiles = [directory / f"profile-{stage}.json" for stage in ("before", "after")]
    if all(path.exists() for path in profiles):
        before, after = [json.loads(path.read_text()) for path in profiles]
        if (
            before["case"] != after["case"]
            or before["request"] != after["request"]
            or before["environment"] != after["environment"]
            or before["environment"] != environment
        ):
            raise ValueError("Different profiling inputs/environments.")
        lines += table(
            "Generation Profile",
            ["Stage", "Before (ms)", "After (ms)"],
            [
                [name, f"{value['stream_ms']:.2f}", f"{after['stages'][name]['stream_ms']:.2f}"]
                for name, value in before["stages"].items()
            ],
        )
        lines += [
            f"Profile: Transformers, {before['case']}. CUDA-event stream intervals include launch/idle gaps.",
            "Instrumentation adds overhead; use the uninstrumented tables above for speed comparisons.",
            "[Before profile](profile-before.json) | [After profile](profile-after.json)",
            "",
        ]
        if "kernels" in before and "kernels" in after:
            lines += table(
                "Profiled Kernel Calls",
                ["Before", "After"],
                [
                    [
                        before["kernels"]["calls"],
                        after["kernels"]["calls"],
                    ]
                ],
            )
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "FAST_LFM_COMPILE_DEPTH=0 python -m benchmarks.compare_backends --output-dir results/optimization/before",
        "python -m benchmarks.compare_backends --output-dir results/optimization/after",
        "FAST_LFM_COMPILE_DEPTH=0 python -m benchmarks.profile --output results/optimization/profile-before.json",
        "python -m benchmarks.profile --output results/optimization/profile-after.json",
        "python -m benchmarks.optimization_report",
        "```",
        "",
        "Set `FAST_LFM_COMPILE_DEPTH=0` to use the previous graph-only depth path.",
        "Compiler API: [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile).",
        "",
    ]
    (directory / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    waveform_path.write_text(json.dumps(waveforms, indent=2) + "\n")
    (directory / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/optimization"))
    write_report(parser.parse_args().directory)
