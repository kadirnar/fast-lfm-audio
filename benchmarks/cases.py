"""Fixed prompts and exact-length audio fixtures shared by both engines."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "tts_uk_female": {
        "prompt": "Perform TTS. Use the UK female voice.",
        "text": "Hello, this is a test of fast audio generation.",
    },
    "tts_us_male": {
        "prompt": "Perform TTS. Use the US male voice.",
        "text": "The train leaves at nine in the morning. Please arrive at the station ten minutes early.",
    },
    "tts_us_female": {
        "prompt": "Perform TTS. Use the US female voice.",
        "text": "A warm cup of tea and a good book make a quiet evening feel special.",
    },
    "tts_uk_male": {
        "prompt": "Perform TTS. Use the UK male voice.",
        "text": "Welcome to the museum. Our next guided tour begins in fifteen minutes near the main entrance.",
    },
    "chat_text": {
        "prompt": "Respond with interleaved text and audio.",
        "text": "Suggest a short, friendly greeting for a coffee shop.",
        "mode": "interleaved",
    },
    "chat_audio": {
        "prompt": "Respond with interleaved text and audio.",
        "audio": "vendor/liquid-audio/assets/question.wav",
        "mode": "interleaved",
    },
}

TTS_TEXT = """
On Saturday morning, Emma opened the windows and listened to the quiet street below.
She had promised to meet her friend at the library, but there was still plenty of
time for breakfast. She filled the kettle, sliced some bread, and placed a small
notebook beside her cup. The notebook contained a list of places she wanted to
visit during the coming year. Some were close enough for an afternoon walk, while
others would require a train ticket and a little more planning.
At half past nine, she put on her coat and stepped outside. The air was cool, and
the pavement was still damp from the rain. A delivery driver was arranging boxes
near the corner shop. Further along the road, two neighbors were discussing the
new community garden. Emma stopped to read the notice on its gate. Volunteers
would meet every Sunday to prepare the soil, plant vegetables, and repair the
wooden benches. She wrote the meeting time in her notebook before continuing.
The library stood beside a small square with a fountain in the middle. Inside,
the reading room was bright and peaceful. Her friend Daniel was already sitting
by the window with a stack of books about architecture. They were planning a
walking tour for a group of visitors and wanted to learn more about the oldest
buildings in town. One book included photographs taken almost a century earlier.
They compared the pictures with a recent map and marked the streets that had
changed the most. The old cinema had become a concert hall, and the former post
office now housed a collection of local paintings.
After an hour of reading, they took a break at a nearby cafe. Daniel ordered a
glass of water and a sandwich. Emma chose soup and another cup of tea.
"""


def make_duration_cases(output_dir, seconds=(5, 20, 100)):
    """Generate TTS prefixes and repeated-speech chat inputs, never fake outputs."""
    word_counts = {5: 40, 20: 100, 100: 300}
    if not seconds or any(duration not in word_counts for duration in seconds):
        raise ValueError("Durations must be 5, 20, or 100 seconds.")
    inputs = output_dir.resolve() / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    source = ROOT / CASES["chat_audio"]["audio"]
    wave, rate = sf.read(source, dtype="float32")
    if wave.ndim != 1 or rate != 16000:
        raise ValueError("Expected the pinned mono 16 kHz upstream question fixture.")
    cases, files = {}, {}
    for duration in seconds:
        frames = math.ceil(duration * 12.5)
        cases[f"tts_{duration}s"] = {
            "prompt": CASES["tts_uk_female"]["prompt"],
            "text": " ".join(TTS_TEXT.split()[: word_counts[duration]]),
            "task": "tts",
            "target_seconds": duration,
            "target_frames": frames,
            "max_new_tokens": frames + 1,
            "duration_method": "Real autoregressive prefix; one audio-start token plus N audio frames. EOS unchanged.",
        }
    for duration in seconds:
        path = inputs / f"question_{duration}s.wav"
        sf.write(path, np.resize(wave, duration * rate), rate, subtype="PCM_16")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases[f"chat_{duration}s"] = {
            "prompt": CASES["chat_audio"]["prompt"],
            "audio": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "mode": "interleaved",
            "task": "chat",
            "input_seconds": duration,
            "max_new_tokens": 512,
            "duration_method": "Upstream question tiled and cropped to exact input length; synthetic repeated speech.",
        }
    provenance = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_seconds": len(wave) / rate,
        "sampling_rate": rate,
        "seconds": list(seconds),
        "files": files,
    }
    (output_dir / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (output_dir / "inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return cases
