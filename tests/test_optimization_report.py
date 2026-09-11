import json

import numpy as np
import pytest
import soundfile as sf

from benchmarks.cases import ROOT
from benchmarks.compare_backends import BACKENDS
from benchmarks.optimization_report import compare_waveforms, write_report
from benchmarks.profile import kernel_summary


@pytest.fixture
def measurements(tmp_path):
    for stage in ("before", "after"):
        (tmp_path / stage).mkdir()
        for engine in BACKENDS:
            data = json.loads((ROOT / "results/backends/inference" / f"{engine}.json").read_text())
            data["cases"] = {"tts_5s": data["cases"]["tts_5s"]}
            if stage == "after":
                for sample in data["cases"]["tts_5s"]["samples"]:
                    sample["total_ms"] /= 2
                data["cases"]["tts_5s"]["median"]["total_ms"] /= 2
            (tmp_path / stage / f"{engine}.json").write_text(json.dumps(data))
    return tmp_path


def test_optimization_report_requires_exact_outputs_for_speedups(measurements):
    write_report(measurements)
    report = (measurements / "REPORT.md").read_text()
    assert report.count("2.00x") == 3
    path = measurements / "after/sglang.json"
    data = json.loads(path.read_text())
    data["cases"]["tts_5s"]["hashes"]["audio_codes"] = "changed"
    path.write_text(json.dumps(data))
    write_report(measurements)
    report = (measurements / "REPORT.md").read_text()
    assert report.count("2.00x") == 2
    assert "different output" in report


def test_optimization_report_rejects_environment_changes(measurements):
    path = measurements / "after/fast.json"
    data = json.loads(path.read_text())
    data["max_cache_len"] = 2048
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Different environments"):
        write_report(measurements)


def test_optimization_report_rejects_stale_medians(measurements):
    path = measurements / "after/fast.json"
    data = json.loads(path.read_text())
    data["cases"]["tts_5s"]["median"]["total_ms"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid recorded median"):
        write_report(measurements)


def test_profile_counts_kernels_without_double_counting_cpu_annotations(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "cpu_op", "name": "depthformer", "dur": 9000},
                    {"cat": "kernel", "name": "gemv", "dur": 1000},
                    {"cat": "kernel", "name": "gemv", "dur": 1500},
                ]
            }
        )
    )
    summary = kernel_summary(path)
    assert summary["calls"] == 2 and summary["ms"] == 2.5
    assert summary["top"] == {"gemv": {"calls": 2, "ms": 2.5}}


def test_waveform_comparison_is_separate_from_token_parity(tmp_path):
    before, after = tmp_path / "before.wav", tmp_path / "after.wav"
    wave = np.array([0.1, 0.2, 0.3])
    sf.write(before, wave, 24000)
    sf.write(after, wave, 24000)
    assert compare_waveforms(before, after)["exact"]
    sf.write(after, wave + 0.01, 24000)
    check = compare_waveforms(before, after)
    assert not check["exact"] and check["snr_db"] > 20
