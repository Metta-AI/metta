"""Policy architecture configurations for recipe experiments.
This module provides a centralized registry of policy architectures
that can be used across recipe files via string-based selection.
"""

from metta.agent.policies.trxl import TRXLConfig
from metta.agent.policies.vit import ViTDefaultConfig

ARCHITECTURES = {
    "default": ViTDefaultConfig(),
    "trxl": TRXLConfig(),
}
