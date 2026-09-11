from pathlib import Path
from unittest.mock import Mock

import pytest

from fast_lfm_audio import Pipeline
from fast_lfm_audio.pipeline import prepare_inputs


@pytest.fixture
def pipeline():
    pipeline = object.__new__(Pipeline)
    pipeline.processor = Mock()
    return pipeline


def test_request_preparation_for_text_and_audio(pipeline):
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


@pytest.mark.parametrize(
    "inputs, mode",
    [
        ({"text": "Hello."}, "sequential"),
        ({"task": "chat", "text": "Hello."}, "interleaved"),
        ({"task": "chat", "audio": "question.wav"}, "interleaved"),
        ({"task": "asr", "audio": "question.wav"}, "sequential"),
    ],
)
def test_one_call_pipeline_uses_task_defaults(pipeline, inputs, mode):
    pipeline.generate = Mock(return_value=("answer", object(), object()))
    result = pipeline(**inputs)
    pipeline.generate.assert_called_once_with(
        pipeline.processor.apply_chat_template.return_value,
        generation_mode=mode,
        max_new_tokens=512,
        text_top_k=1,
        audio_top_k=1,
        audio_temperature=0.0,
    )
    assert result is pipeline.generate.return_value


def test_one_call_pipeline_keeps_custom_generation_options(pipeline):
    pipeline.generate = Mock()
    pipeline(text="Hello.", voice="US male", max_new_tokens=64, audio_top_k=4, audio_temperature=0.8)
    pipeline.generate.assert_called_once_with(
        pipeline.processor.apply_chat_template.return_value,
        generation_mode="sequential",
        max_new_tokens=64,
        text_top_k=1,
        audio_top_k=4,
        audio_temperature=0.8,
    )
    messages = pipeline.processor.apply_chat_template.call_args.args[0]
    assert messages[0]["content"] == "Perform TTS. Use the US male voice."


def test_invalid_inputs_fail_before_processor_or_model_loading():
    processor = Mock()
    for inputs in ({}, {"text": "hi", "audio": "question.wav"}, {"text": "  "}):
        with pytest.raises(ValueError):
            prepare_inputs(processor, prompt="Perform TTS.", **inputs)
    processor.apply_chat_template.assert_not_called()
    with pytest.raises(ValueError, match="codec"):
        Pipeline(codec="unknown")
