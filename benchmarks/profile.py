"""Locate generation bottlenecks after warmup; profiling is not a latency benchmark."""

import argparse
import json
import statistics
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

from .benchmark import Runner, metadata
from .cases import ROOT


def kernel_summary(path):
    with path.open() as stream:
        trace = json.load(stream)
    totals = defaultdict(lambda: {"calls": 0, "ms": 0.0})
    for event in trace["traceEvents"]:
        if event.get("cat") == "kernel":
            totals[event["name"]]["calls"] += 1
            totals[event["name"]]["ms"] += event["dur"] / 1000
    return {
        "calls": sum(item["calls"] for item in totals.values()),
        "ms": sum(item["ms"] for item in totals.values()),
        "top": dict(sorted(totals.items(), key=lambda item: item[1]["ms"], reverse=True)[:10]),
    }


@contextmanager
def measure_stages(model, events):
    originals = []

    def wrap(owner, attribute, label):
        original = getattr(owner, attribute)
        originals.append((owner, attribute, original))

        def measured(*args, **kwargs):
            name = label
            if label == "backbone":
                name = "llm_prefill" if kwargs.get("past_key_values") is None else "llm_decode"
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            with torch.profiler.record_function(name):
                start.record()
                result = original(*args, **kwargs)
                end.record()
            events[name].append((start, end))
            return result

        setattr(owner, attribute, measured)

    wrap(model.model, "_prepare_inputs_embeds", "input_embeddings")
    wrap(model.model.lfm, "forward", "backbone")
    wrap(model, "_sample_audio_frame", "depthformer")
    try:
        yield
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="tts_5s")
    parser.add_argument("--output", type=Path, default=ROOT / "results/optimization/profile.json")
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    case = json.loads((ROOT / "results/durations/cases.json").read_text())[args.case]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    runner = Runner("fast", 4096)
    runner.run(case, case["max_new_tokens"])
    samples = [runner.run(case, case["max_new_tokens"])[0] for _ in range(5)]
    events = defaultdict(list)
    with measure_stages(runner.model, events):
        timing, output, _ = runner.run(case, case["max_new_tokens"])
    stages = {
        name: {"calls": len(pairs), "stream_ms": sum(start.elapsed_time(end) for start, end in pairs)}
        for name, pairs in events.items()
    }
    report = {
        "environment": metadata(),
        "case": args.case,
        "request": case,
        "uninstrumented_median_ms": {key: statistics.median(s[key] for s in samples) for key in samples[0]},
        "profiled_ms": timing,
        "stages": stages,
        "events": output["modalities"].numel(),
        "note": "CUDA-event stream intervals can include launch/idle gaps; they are not pure kernel times.",
    }
    if args.trace:
        with (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile,
            measure_stages(runner.model, defaultdict(list)),
        ):
            runner.run(case, case["max_new_tokens"])
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))
        report["kernels"] = kernel_summary(args.trace)
        print(profile.key_averages().table(sort_by="self_device_time_total", row_limit=25), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
