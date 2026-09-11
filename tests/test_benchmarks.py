import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from benchmarks.cases import ROOT, make_duration_cases
from benchmarks.report import compare_outputs, waveform_statistics, write_report


def test_duration_fixtures_and_generation_budgets(tmp_path):
    cases = make_duration_cases(tmp_path)
    source, rate = sf.read(ROOT / "vendor/liquid-audio/assets/question.wav", dtype="float32")
    recorded = json.loads((ROOT / "results/durations/cases.json").read_text())
    provenance = json.loads((tmp_path / "inputs.json").read_text())
    for seconds, frames in ((5, 63), (20, 250), (100, 1250)):
        tts = cases[f"tts_{seconds}s"]
        assert tts["target_frames"] == frames
        assert tts["max_new_tokens"] == frames + 1
        assert tts == recorded[f"tts_{seconds}s"]
        chat = cases[f"chat_{seconds}s"]
        assert chat["max_new_tokens"] == 512
        path = Path(chat["audio"])
        audio, sample_rate = sf.read(path, dtype="float32")
        assert sample_rate == rate == 16000
        assert len(audio) == seconds * rate
        np.testing.assert_array_equal(audio, np.resize(source, seconds * rate))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["files"][path.name]
    assert json.loads((tmp_path / "cases.json").read_text()) == cases


def test_waveform_audit_detects_a_quiet_tail(tmp_path):
    path = tmp_path / "quiet.wav"
    wave = np.concatenate((np.full(24000, 0.1), np.zeros(6 * 24000)))
    sf.write(path, wave, 24000)
    stats = waveform_statistics(path)
    assert stats["seconds"] == 7
    assert stats["quiet_seconds"] == stats["longest_quiet_seconds"] == stats["trailing_quiet_seconds"] == 6
    assert stats["finite"]


def test_parity_checks_tensors_even_when_recorded_hashes_match():
    case = {"events": 3, "audio_frames": 2, "repeat_exact": True}
    expected = {key: torch.zeros(2, dtype=torch.long) for key in ("sequences", "audio_codes", "modalities")}
    actual = {key: value.clone() for key, value in expected.items()}
    assert compare_outputs(case, case, expected, actual)["all_exact"]
    actual["audio_codes"][0] = 1
    result = compare_outputs(case, case, expected, actual)
    assert result["method"] == "tensors"
    assert not result["checks"]["audio_codes"]
    assert not result["all_exact"]


def test_report_from_published_json_and_mismatched_requests(tmp_path):
    for engine in ("liquid", "fast"):
        data = json.loads((ROOT / "results" / f"{engine}.json").read_text())
        data["cases"] = {"tts_uk_female": data["cases"]["tts_uk_female"]}
        (tmp_path / f"{engine}.json").write_text(json.dumps(data))
    write_report(tmp_path)
    parity = json.loads((tmp_path / "parity.json").read_text())
    assert parity["tts_uk_female"]["fast"]["method"] == "recorded_sha256"
    assert parity["tts_uk_female"]["fast"]["all_exact"]
    data["cases"]["tts_uk_female"]["request"]["text"] = "A different prompt."
    (tmp_path / "fast.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Different requests"):
        write_report(tmp_path)
