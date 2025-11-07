"""PufferLib Core - Minimal vectorized environment functionality."""

import sys


def _import_modules():
    import pufferlib

    current_module = sys.modules[__name__]
    current_module.PufferEnv = pufferlib.pufferlib.PufferEnv
    current_module.set_buffers = pufferlib.pufferlib.set_buffers
    current_module.unroll_nested_dict = pufferlib.pufferlib.unroll_nested_dict
    current_module.APIUsageError = pufferlib.pufferlib.APIUsageError

    try:

        current_module._C = pufferlib._C
    except ImportError:
        pass

    current_module.pytorch = pufferlib.pytorch
    current_module.models = pufferlib.models

    current_module.pufferl = pufferlib.pufferl

    return pufferlib.spaces, pufferlib.pufferlib, pufferlib.emulation, pufferlib.vector, pufferlib.pytorch, pufferlib.models, pufferlib.pufferl


spaces, pufferlib, emulation, vector, pytorch, models, pufferl = _import_modules()

__version__ = "3.0.3"
__all__ = [
    "APIUsageError",
    "PufferEnv",
    "emulation",
    "models",
    "pufferl",
    "pufferlib",
    "pytorch",
    "set_buffers",
    "spaces",
    "unroll_nested_dict",
    "vector",
]
