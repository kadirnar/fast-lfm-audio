"""Compare decoders on the very same eight-codebook sequences."""

import argparse
import json
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch
from liquid_audio import LFM2AudioProcessor

from fast_lfm_audio.pipeline import MimiDecoder, decodable_codes, model_path


def measure(function, codes, repeats):
    torch.cuda.synchronize()
    start = time.perf_counter()
    function(codes)
    torch.cuda.synchronize()
    cold = (time.perf_counter() - start) * 1000
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        wave = function(codes)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    if not bool(torch.isfinite(wave).all()):
        raise RuntimeError("Decoder produced non-finite samples")
    return {"cold_ms": cold, "median_ms": statistics.median(samples), "samples_ms": samples}, wave.clone()


def fidelity(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f"Waveform shape mismatch: {actual.shape} vs {reference.shape}")
    error = actual.float() - reference.float()
    return {
        "max_abs_error": error.abs().max().item(),
        "rmse": error.square().mean().sqrt().item(),
        "snr_db": (
            10 * torch.log10(reference.float().square().mean() / error.square().mean().clamp_min(1e-30))
        ).item(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokens = torch.load(args.output_dir / "liquid_tokens.pt", weights_only=True)
    processor = LFM2AudioProcessor.from_pretrained(Path(model_path())).eval()
    reference = MimiDecoder(optimized=False)
    fast = MimiDecoder()
    report = {"repeats": args.repeats, "cases": {}}
    for name, output in tokens.items():
        codes = decodable_codes(output["audio_codes"].cuda())
        if not codes.shape[-1]:
            continue
        case = {"frames": codes.shape[-1], "audio_seconds": codes.shape[-1] / 12.5}
        case["liquid_detokenizer"], liquid_wave = measure(processor.decode, codes, args.repeats)
        case["mimi_transformers_fp32"], reference_wave = measure(reference, codes, args.repeats)
        case["fast_mimi"], fast_wave = measure(fast, codes, args.repeats)
        fast.bucketed = False
        case["fast_mimi_unbucketed"], unbucketed_wave = measure(fast, codes, args.repeats)
        fast.bucketed = True
        case["padding_vs_unpadded"] = fidelity(fast_wave, unbucketed_wave)
        case["fast_vs_mimi_fp32"] = fidelity(fast_wave, reference_wave)
        case["unbucketed_vs_mimi_fp32"] = fidelity(unbucketed_wave, reference_wave)
        case["padding_snr_degradation_db"] = (
            case["unbucketed_vs_mimi_fp32"]["snr_db"] - case["fast_vs_mimi_fp32"]["snr_db"]
        )
        # The half-precision kernels change reduction order with shape. Check
        # relative signal error, rather than FP32 pointwise tolerances near zero.
        if case["fast_vs_mimi_fp32"]["snr_db"] < 40 or case["padding_snr_degradation_db"] > 1:
            raise RuntimeError(f"Half-precision decoder fidelity gate failed: {name}: {case}")
        case["liquid_detokenizer_vs_mimi"] = fidelity(liquid_wave, reference_wave)
        case["mimi_speedup"] = case["mimi_transformers_fp32"]["median_ms"] / case["fast_mimi"]["median_ms"]
        case["default_decoder_speedup"] = (
            case["liquid_detokenizer"]["median_ms"] / case["fast_mimi"]["median_ms"]
        )
        report["cases"][name] = case
        sf.write(args.output_dir / f"mimi_fp32_{name}.wav", reference_wave[0].cpu().numpy(), 24000)
        sf.write(args.output_dir / f"fast_mimi_same_tokens_{name}.wav", fast_wave[0].cpu().numpy(), 24000)
        (args.output_dir / "codec.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            name,
            f"Mimi {case['mimi_speedup']:.2f}x, default decoder {case['default_decoder_speedup']:.2f}x, "
            f"decode SNR {case['fast_vs_mimi_fp32']['snr_db']:.1f} dB",
            flush=True,
        )


if __name__ == "__main__":
    main()
