"""PufferLib Core - Minimal vectorized environment functionality."""

import sys


def _import_modules():
    from . import pufferlib, spaces

    current_module = sys.modules[__name__]
    current_module.PufferEnv = pufferlib.PufferEnv
    current_module.set_buffers = pufferlib.set_buffers
    current_module.unroll_nested_dict = pufferlib.unroll_nested_dict
    current_module.APIUsageError = pufferlib.APIUsageError

    from . import emulation, vector

    try:
        from . import _C

        current_module._C = _C
    except ImportError:
        pass

    from . import models, pytorch

    current_module.pytorch = pytorch
    current_module.models = models

    from . import pufferl

    current_module.pufferl = pufferl

    return spaces, pufferlib, emulation, vector, pytorch, models, pufferl


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
