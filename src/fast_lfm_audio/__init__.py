import os

# Spawned SGLang workers import the caller before loading SGLang's config registry.
if os.environ.get("FAST_LFM_NATIVE_BACKEND") == "sglang":
    from .backends.compat import configure_sglang

    configure_sglang()

from .pipeline import Pipeline
from .runtime import optimize
from .strict import fp32_precision

__all__ = ["Pipeline", "fp32_precision", "optimize"]
