"""One entry point for standard and duration benchmarks; one GPU process at a time."""

import argparse
import subprocess
import sys
from pathlib import Path

from .cases import CASES, ROOT, make_duration_cases
from .report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("standard", "durations"), default="standard")
    parser.add_argument(
        "--engines", nargs="+", choices=("liquid", "hf", "depth", "fast"), default=["liquid", "fast"]
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seconds", nargs="+", type=int, choices=(5, 20, 100), default=[5, 20, 100])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--codec-repeats", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-cache-len", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not {"liquid", "fast"}.issubset(args.engines) or len(set(args.engines)) != len(args.engines):
        parser.error("Select liquid and fast, with no duplicate engines.")
    if min(args.repeats, args.codec_repeats, args.max_new_tokens) < 1:
        parser.error("Repeat counts and token budget must be positive.")
    durations = args.suite == "durations"
    output_dir = (args.output_dir or ROOT / "results" / ("durations" if durations else "")).resolve()
    if args.report_only:
        write_report(output_dir, args.engines)
        return
    cases = make_duration_cases(output_dir, args.seconds) if durations else CASES
    selected = args.cases or list(cases)
    if any(name not in cases for name in selected):
        parser.error(f"Choose cases from: {', '.join(cases)}")
    cache_len = args.max_cache_len if args.max_cache_len is not None else (4096 if durations else 2048)
    if cache_len < 16:
        parser.error("--max-cache-len must be at least 16")
    for engine in args.engines:
        extra = ["--cases-file", str(output_dir / "cases.json")] if durations else []
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.benchmark",
                "--engine",
                engine,
                "--repeats",
                str(args.repeats),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--max-cache-len",
                str(cache_len),
                "--output-dir",
                str(output_dir),
                *extra,
                "--cases",
                *selected,
            ],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.codec",
            "--output-dir",
            str(output_dir),
            "--repeats",
            str(args.codec_repeats),
        ],
        cwd=ROOT,
        check=True,
    )
    write_report(output_dir, args.engines)


if __name__ == "__main__":
    main()
