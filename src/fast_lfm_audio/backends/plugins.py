"""Opt-in engine plugins. Installing this package does not replace text models."""

import os


def register_vllm():
    if os.environ.get("FAST_LFM_NATIVE_BACKEND") == "vllm":
        from vllm import ModelRegistry

        ModelRegistry.register_model("Lfm2ForCausalLM", "fast_lfm_audio.backends.vllm:NativeAudioLfm2")


def register_sglang():
    if os.environ.get("FAST_LFM_NATIVE_BACKEND") != "sglang":
        return
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    class AudioScheduler(Scheduler):
        def reset_lfm_audio(self, mode):
            self.tp_worker.model_runner.model.audio_state.reset(mode)

    def sample_audio(original, runner, logits_output, forward_batch):
        state = getattr(runner.model, "audio_state", None)
        if state is None:
            return original(runner, logits_output, forward_batch)
        logits_output.customized_info = None
        if state.active:
            logits_output.next_token_logits, event = state.step(
                logits_output.hidden_states[-1], logits_output.next_token_logits
            )
            logits_output.customized_info = {"lfm_audio_events": [event]}
        return original(runner, logits_output, forward_batch)

    HookRegistry.register("sglang.srt.managers.scheduler.Scheduler", AudioScheduler, HookType.REPLACE)
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.sample", sample_audio, HookType.AROUND
    )
