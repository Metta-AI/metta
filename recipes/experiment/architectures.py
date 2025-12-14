"""Central registry for policy architecture short names used in recipes.

We store class paths (or callables) so heavy modules load lazily, and expose
helpers to fetch a fresh PolicyArchitecture by short name.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from metta.agent.policy import PolicyArchitecture
from mettagrid.util.module import load_symbol

# Short names -> class paths (or factories). Keep this list small and focused on
# architectures referenced by recipes.
_ARCHITECTURE_SPECS: dict[str, str | Callable[[], PolicyArchitecture]] = {
    "vit": "metta.agent.policies.vit.ViTDefaultConfig",
    "trxl": "metta.agent.policies.trxl.TRXLConfig",
    "fast": "metta.agent.policies.fast.FastConfig",
    "fast_dynamics": "metta.agent.policies.fast_dynamics.FastDynamicsConfig",
    "memory_free": "metta.agent.policies.memory_free.MemoryFreeConfig",
    "agalite": "metta.agent.policies.agalite.AGaLiTeConfig",
    "puffer": "metta.agent.policies.puffer.PufferPolicyConfig",
}


def architecture_names() -> list[str]:
    return sorted(_ARCHITECTURE_SPECS)


@lru_cache(None)
def _resolve(name: str) -> PolicyArchitecture:
    spec = _ARCHITECTURE_SPECS[name]
    cls_or_factory = load_symbol(spec) if isinstance(spec, str) else spec
    return cls_or_factory()


def get_architecture(name: str) -> PolicyArchitecture:
    """Return a PolicyArchitecture from a short name or dotted spec string."""
    if name in _ARCHITECTURE_SPECS:
        return _resolve(name).model_copy(deep=True)
    # Allow fully qualified specs to keep advanced usage flexible.
    return PolicyArchitecture.from_spec(name)
