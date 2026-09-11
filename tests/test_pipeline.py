from pathlib import Path
from unittest.mock import Mock

import pytest

from fast_lfm_audio import Pipeline
from fast_lfm_audio.pipeline import prepare_inputs


def test_request_preparation_for_text_and_audio():
    pipeline = object.__new__(Pipeline)
    pipeline.processor = Mock()
    pipeline.prepare(text="Hello.", voice="US female")
    args, kwargs = pipeline.processor.apply_chat_template.call_args
    assert args[0] == [
        {"role": "system", "content": "Perform TTS. Use the US female voice."},
        {"role": "user", "content": "Hello."},
    ]
    assert kwargs["processor_kwargs"] == {"audio_kwargs": {"device": "cuda"}}
    pipeline.prepare(task="chat", audio=Path("question.wav"))
    messages = pipeline.processor.apply_chat_template.call_args.args[0]
    assert messages[0]["content"] == "Respond with interleaved text and audio."
    assert messages[1]["content"] == [{"type": "audio", "path": "question.wav"}]


def test_invalid_inputs_fail_before_processor_or_model_loading():
    processor = Mock()
    for inputs in ({}, {"text": "hi", "audio": "question.wav"}, {"text": "  "}):
        with pytest.raises(ValueError):
            prepare_inputs(processor, prompt="Perform TTS.", **inputs)
    processor.apply_chat_template.assert_not_called()
    with pytest.raises(ValueError, match="codec"):
        Pipeline(codec="unknown")
