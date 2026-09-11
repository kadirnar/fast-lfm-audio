"""One report format for both suites, with timing, parity, and waveform checks."""

import json

import numpy as np
import soundfile as sf
import torch

OUTPUTS = ("sequences", "audio_codes", "modalities")


def compare_outputs(reference, candidate, reference_tokens=None, candidate_tokens=None):
    """Use tensors locally; published JSON can be checked using recorded hashes."""
    tensors_available = reference_tokens is not None and candidate_tokens is not None
    checks = {
        key: torch.equal(reference_tokens[key], candidate_tokens[key])
        if tensors_available
        else reference["hashes"][key] == candidate["hashes"][key]
        for key in OUTPUTS
    }
    checks["counts"] = all(reference[key] == candidate[key] for key in ("events", "audio_frames"))
    checks["repeats_exact"] = reference["repeat_exact"] and candidate["repeat_exact"]
    return {
        "method": "tensors" if tensors_available else "recorded_sha256",
        "checks": checks,
        "all_exact": all(checks.values()),
    }


def waveform_statistics(path):
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim != 1:
        raise ValueError("Expected mono benchmark output.")
    seconds = len(wave) // rate
    rms = np.sqrt(np.mean(wave[: seconds * rate].reshape(seconds, rate) ** 2, axis=1))
    quiet = rms < 0.001
    longest = current = 0
    for value in quiet:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "seconds": len(wave) / rate,
        "finite": bool(np.isfinite(wave).all()),
        "peak": float(np.abs(wave).max(initial=0)),
        "near_full_scale_fraction": float((np.abs(wave) >= 0.9999).mean()) if wave.size else 0.0,
        "quiet_threshold_rms": 0.001,
        "quiet_seconds": int(quiet.sum()),
        "longest_quiet_seconds": longest,
        "trailing_quiet_seconds": current,
        "one_second_rms": rms.tolist(),
    }


def table(title, columns, rows):
    return [
        f"## {title}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        "",
    ]


def write_report(output_dir, engines=("liquid", "fast")):
    data = {engine: json.loads((output_dir / f"{engine}.json").read_text()) for engine in engines}
    cases = data["liquid"]["cases"]
    environment = data["liquid"]["environment"]
    tokens = {}
    for engine in engines:
        path = output_dir / f"{engine}_tokens.pt"
        tokens[engine] = torch.load(path, weights_only=True) if path.exists() else {}
        if set(data[engine]["cases"]) != set(cases):
            raise ValueError(f"Different case sets: {engine}")
        for key in (
            "gpu",
            "torch",
            "model_revision",
            "commits",
            "cpu_threads",
            "dtype",
            "text_top_k",
            "audio_top_k",
        ):
            if data[engine]["environment"][key] != environment[key]:
                raise ValueError(f"Different benchmark environments: {engine}: {key}")
    checks = {}
    for name, reference in cases.items():
        checks[name] = {}
        for engine in engines:
            candidate = data[engine]["cases"][name]
            if reference["request"] != candidate["request"]:
                raise ValueError(f"Different requests: {engine}: {name}")
            if len(reference["samples"]) != len(candidate["samples"]):
                raise ValueError(f"Different repeat counts: {engine}: {name}")
            if engine != "liquid":
                checks[name][engine] = compare_outputs(
                    reference, candidate, tokens["liquid"].get(name), tokens[engine].get(name)
                )
    (output_dir / "parity.json").write_text(json.dumps(checks, indent=2) + "\n")

    waveforms_path = output_dir / "waveforms.json"
    saved = json.loads(waveforms_path.read_text()) if waveforms_path.exists() else {}
    waveforms = {name: saved.get(name, {}) for name in cases}
    for name in cases:
        for engine in (*engines, "mimi_fp32"):
            path = output_dir / f"{engine}_{name}.wav"
            if path.exists():
                waveforms[name][engine] = waveform_statistics(path)
    waveforms_path.write_text(json.dumps(waveforms, indent=2) + "\n")

    repeats = len(next(iter(cases.values()))["samples"])
    lines = [
        "# LFM2.5 Audio Benchmark",
        "",
        f"{environment['gpu']}; PyTorch {environment['torch']}; CUDA {environment['cuda']}; Python {environment['python']}.",
        f"Batch one, BF16 model, greedy sampling, one first request and {repeats} warm repeats per case.",
        "Engines run sequentially in separate processes. Tables show synchronized wall-clock medians; model loading is excluded.",
        "Input preparation, token generation, and full waveform decoding are included. Every request is generated again.",
        "Liquid Audio uses its FP32 LFM detokenizer; fast uses half-precision fast-mimi. Token parity is not waveform parity.",
        "",
    ]
    if any("target_frames" in case["request"] for case in cases.values()):
        lines += [
            "TTS duration cases generate prefixes of longer text: 63/250/1250 frames give 5.04/20/100 s. EOS is unchanged.",
            "They stop at an event budget, not natural sentence completion. The 100 s fixture uses 300 words; an initial 420-word probe returned text instead of speech.",
            "Chat inputs repeat and crop the upstream 4.904 s question to 5/20/100 s, with a 512-event response budget.",
            "This measures synthetic input-length scaling, not natural long-conversation quality. Output durations are independent of input durations.",
            "",
        ]
    rows = []
    for name, reference in cases.items():
        candidate = data["fast"]["cases"][name]
        base, fast = reference["median"], candidate["median"]
        exact = checks[name]["fast"]["all_exact"]
        rows.append(
            (
                name,
                reference["request"].get("input_seconds", "-"),
                f"{base['audio_seconds']:.2f} / {fast['audio_seconds']:.2f}",
                f"{base['total_ms'] / 1000:.3f}",
                f"{fast['total_ms'] / 1000:.3f}",
                f"{base['total_ms'] / fast['total_ms']:.2f}x" if exact else "not comparable",
                "exact" if exact else "DIFFERS",
                f"{reference['limit_reached']} / {candidate['limit_reached']}",
            )
        )
    lines += table(
        "End-to-End Latency",
        (
            "Case",
            "Input s",
            "Output s, base / fast",
            "Liquid s",
            "Fast s",
            "Speedup",
            "Tokens",
            "Budget reached, base / fast",
        ),
        rows,
    )
    lines += [
        "A budget-limited output is partial. One event is a text token or an eight-codebook audio frame.",
        "",
    ]
    tail = waveforms.get("tts_100s", {}).get("fast", {}).get("trailing_quiet_seconds", 0)
    if tail >= 5:
        lines += [
            f"**100 s TTS warning:** the optimized output has a {tail} s low-signal tail. This is a compute stress test, not 100 s of successful continuous speech. Compare the reference below.",
            "",
        ]

    rows = []
    for name in cases:
        for engine in engines:
            case = data[engine]["cases"][name]
            median = case["median"]
            rows.append(
                (
                    f"{name} / {engine}",
                    *(
                        f"{median.get(key, float('nan')):.1f}"
                        for key in ("prepare_ms", "generation_ms", "decode_ms", "first_audio_token_ms")
                    ),
                    f"{case['cold']['total_ms'] / 1000:.3f}",
                    f"{case['p95_total_ms'] / 1000:.3f}",
                    f"{case['peak_allocated_bytes'] / 2**30:.2f}",
                )
            )
    lines += table(
        "Timing Breakdown",
        (
            "Case / engine",
            "Prepare ms",
            "Generate ms",
            "Decode ms",
            "First audio token ms",
            "First request s",
            "p95 total s",
            "Peak GiB",
        ),
        rows,
    )
    lines += [
        "First audio token is not first playable PCM audio: this pipeline decodes the complete output offline.",
        "First-request timing includes graph capture, but reuses existing Triton caches and any earlier shapes in that process.",
        "Model setup (s): "
        + ", ".join(f"{engine} {data[engine]['setup_seconds']:.3f}" for engine in engines)
        + ".",
        "",
    ]
    rows = [
        (f"{name} / {engine}", signal["quiet_seconds"], signal["trailing_quiet_seconds"], signal["finite"])
        for name, signals in waveforms.items()
        for engine, signal in signals.items()
    ]
    if rows:
        lines += table(
            "Waveform Checks",
            ("Case / decoder", "Low-signal seconds", "Trailing low-signal seconds", "Finite"),
            rows,
        )
        lines += [
            "Low-signal: RMS < 0.001 (-60 dBFS) in full one-second windows of saved PCM16 WAVs. This is not an intelligibility score.",
            "Native FP32 Mimi decodes the same reference codes, separating codec differences from optimization error. Saved checks are reused when WAVs are absent.",
            "",
        ]
    codec_path = output_dir / "codec.json"
    if codec_path.exists():
        codec = json.loads(codec_path.read_text())
        rows = [
            (
                name,
                *(
                    f"{case[key]['median_ms']:.2f}"
                    for key in ("liquid_detokenizer", "mimi_transformers_fp32", "fast_mimi")
                ),
                f"{case['default_decoder_speedup']:.2f}x",
                f"{case['mimi_speedup']:.2f}x",
                f"{case['fast_vs_mimi_fp32']['snr_db']:.1f}",
            )
            for name, case in codec["cases"].items()
            if name in cases
        ]
        lines += table(
            "Same-Code Decoders",
            ("Case", "LFM ms", "Mimi FP32 ms", "Fast Mimi ms", "vs LFM", "vs Mimi", "SNR dB"),
            rows,
        )
        lines += [
            f"Medians of {codec['repeats']} warm runs on identical codes. SNR compares fast-mimi with native FP32 Mimi; it is not a listening test.",
            "",
        ]
    tuning_path = output_dir / "initial_tuning_probe.json"
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text())
        lines += [
            "## Initial Long-Shape Tuning",
            "",
            f"The initial 100 s probe took {tuning['total_ms'] / 1000:.3f} s, including {tuning['decode_ms'] / 1000:.3f} s of decode/autotuning for the 2048-frame bucket.",
            "Smaller-shape caches already existed. Final warm medians exclude this cost. Prewarm required shapes before serving requests.",
            "",
        ]
    lines += ["## Sources and Artifacts", ""]
    lines += [f"- {name}: `{sha}`" for name, sha in environment["commits"].items()]
    lines += [
        f"- Model: `{environment['model_revision']}`",
        "",
        "Engine JSON files contain every timing sample, requests, metadata, memory measurements, and token hashes.",
        "`parity.json` records whether comparisons used local tensors or published SHA256 hashes. `waveforms.json` records signal checks.",
        "WAV/PT artifacts are generated locally and are not committed. Reproduce with `python -m benchmarks.run`; add `--suite durations` for 5/20/100 s tests.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    print(f"Report: {output_dir / 'REPORT.md'}")
    if not all(result["all_exact"] for case in checks.values() for result in case.values()):
        raise RuntimeError("Token parity failed; see parity.json.")
