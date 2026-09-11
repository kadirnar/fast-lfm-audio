"""Import compatibility for SGLang 0.5.19 with the newer Transformers audio PR."""


def configure_sglang():
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    original = AutoConfig.register
    if getattr(original, "_fast_lfm_audio", False):
        return

    def register(model_type, config, exist_ok=False):
        # Permit only SGLang's own duplicate config registrations, not arbitrary collisions.
        if config.__module__.startswith("sglang.") and model_type in CONFIG_MAPPING:
            exist_ok = True
        return original(model_type, config, exist_ok=exist_ok)

    register._fast_lfm_audio = True
    AutoConfig.register = register
