"""PufferLib Core - Minimal vectorized environment functionality."""

from . import pufferlib as core_pufferlib

PufferEnv = core_pufferlib.PufferEnv
set_buffers = core_pufferlib.set_buffers
unroll_nested_dict = core_pufferlib.unroll_nested_dict
APIUsageError = core_pufferlib.APIUsageError
pufferlib = core_pufferlib

from . import spaces
from . import emulation
from . import vector
from . import pytorch
from . import models
from . import pufferl

try:  # pragma: no cover - optional C extensions
    from . import _C  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _C = None  # type: ignore[assignment]

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
