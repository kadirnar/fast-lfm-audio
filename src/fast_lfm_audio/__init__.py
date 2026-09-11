import os

# Spawned SGLang workers import the caller before loading SGLang's config registry.
if os.environ.get("FAST_LFM_NATIVE_BACKEND") == "sglang":
    from .backends.compat import configure_sglang

    configure_sglang()

from .pipeline import Pipeline
from .runtime import optimize

__all__ = ["Pipeline", "optimize"]
