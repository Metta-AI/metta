"""Scripted agent policies for CoGames.

Contains two baseline policies:
- BaselinePolicy: Single/multi-agent resource gathering and heart assembly
- UnclippingPolicy: Extends baseline with extractor unclipping capability

And their hyperparameter configurations:
- BaselineHyperparameters: Configuration for baseline agent behavior
- UnclippingHyperparameters: Configuration for unclipping agent behavior
"""

from cogames.policy.scripted_agent.baseline_agent import (
    BASELINE_HYPERPARAMETER_PRESETS,
    BaselineHyperparameters,
    BaselinePolicy,
)
from cogames.policy.scripted_agent.unclipping_agent import UnclippingHyperparameters, UnclippingPolicy

__all__ = [
    "BaselinePolicy",
    "BaselineHyperparameters",
    "BASELINE_HYPERPARAMETER_PRESETS",
    "UnclippingPolicy",
    "UnclippingHyperparameters",
]
