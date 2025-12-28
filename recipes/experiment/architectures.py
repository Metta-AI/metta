"""Central registry for policy architecture short names used in recipes.

We store class paths (or callables) so heavy modules load lazily, and expose
helpers to fetch a fresh PolicyArchitecture by short name.
"""

from __future__ import annotations

from metta.agent.policy import PolicyArchitecture
from mettagrid.util.module import load_symbol

# Short names -> class paths. Keep this list small and focused on architectures
# referenced by recipes.
_ARCHITECTURE_SPECS: dict[str, str] = {
    # ViT variants
    "vit": "metta.agent.policies.vit.ViTDefaultConfig",
    "vit_size2": "metta.agent.policies.vit_size_2.ViTSize2Config",
    "vit_grpo": "metta.agent.policies.vit_grpo.ViTGRPOConfig",
    "vit_quantile": "metta.agent.policies.vit_quantile.ViTQuantileConfig",
    "vit_reset": "metta.agent.policies.vit_reset.ViTResetConfig",
    # Transformers / memory
    "trxl": "metta.agent.policies.trxl.TRXLConfig",
    "fast": "metta.agent.policies.fast.FastConfig",
    "fast_dynamics": "metta.agent.policies.fast_dynamics.FastDynamicsConfig",
    "memory_free": "metta.agent.policies.memory_free.MemoryFreeConfig",
    "puffer": "metta.agent.policies.puffer.PufferPolicyConfig",
    "cortex": "metta.agent.policies.cortex.CortexBaseConfig",
    # Alternative stacks
    "agalite": "metta.agent.policies.agalite.AGaLiTeConfig",
    "mamba": "metta.agent.policies.mamba_sliding.MambaSlidingConfig",
    "drama": "metta.agent.policies.drama_policy.DramaPolicyConfig",
    "smollm": "metta.agent.policies.smollm.SmolLLMConfig",
    # HRM variants
    "hrm": "metta.agent.policies.hrm.HRMPolicyConfig",
    "hrm_tiny": "metta.agent.policies.hrm.HRMTinyConfig",
}


def architecture_names() -> list[str]:
    return sorted(_ARCHITECTURE_SPECS)


def get_architecture(name: str) -> PolicyArchitecture:
    """Return a PolicyArchitecture from a short name or dotted spec string."""
    if name in _ARCHITECTURE_SPECS:
        cls = load_symbol(_ARCHITECTURE_SPECS[name])
        return cls().model_copy(deep=True)
    # Allow fully qualified specs to keep advanced usage flexible.
    return PolicyArchitecture.from_spec(name)
