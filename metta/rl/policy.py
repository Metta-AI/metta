"""
DEPRECATED: This module is deprecated and will be removed in a future version.
Please use metta.agent.policy_store.build_pytorch_policy instead of load_policy.
"""

import warnings

from metta.agent.policy_store import build_pytorch_policy

warnings.warn(
    "metta.rl.policy module is deprecated. Use metta.agent.policy_store.build_pytorch_policy instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Maintain backward compatibility
load_policy = build_pytorch_policy

# Re-export PytorchAgent for backward compatibility
from metta.agent.pytorch_policy import PytorchPolicy as PytorchAgent

__all__ = ["load_policy", "PytorchAgent"]
